"""Execution gates tested with mocked process, power and clock APIs only."""
import json
import copy
import inspect
import io
from pathlib import Path
import sys
import tempfile
import hashlib
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
try:
    import run_msb_role_mechanism as runner
except ModuleNotFoundError:
    runner=None
from msb_synthetic_helpers import make_prepared_fixture
from test_msb_execution_safety import FakeAPI


class FakeGuard:
    def __init__(self,**kwargs): self.released=False
    def __enter__(self): return self
    def __exit__(self,*args): self.released=True
    def check(self): return 0.
    def evidence(self): return dict(power_request_acquired=True,power_request_released=self.released)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner,"MSB gated runner missing")
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        api=FakeAPI(); api.current_pid=lambda:7; api.current_identity=lambda:(7,99)
        api.awake_guard=FakeGuard
        native=patch.object(runner,"NativeMSBAPI",return_value=api)
        native.start(); self.addCleanup(native.stop)

    def simulate(self,status="COMPLETE",*,final_status=None,tamper=False,guard_factory=FakeGuard,folds=None,
                 allow_fits=False,analysis_callback=None,pre_fit_mutation=None,registration_exhausted=False,
                 expire_export=None,registered_path=None):
        jobs=[]; clock=[0.]; self.fit_calls=[]; self.load_calls=[]
        def monotonic(): clock[0]+=.2; return clock[0]
        def api_factory():
            api=FakeAPI(); api.tick_ms=lambda:int(monotonic()*1000)
            api.current_pid=lambda:7; api.awake_guard=guard_factory
            api.current_identity=lambda:(7,99)
            return api
        parent_api=api_factory()
        class Parent:
            def __init__(inner,budget,run_id,digest,*,api=None): inner.budget=budget; inner.guard=guard_factory()
            def __enter__(inner): inner.guard.__enter__(); return inner
            def __exit__(inner,*args): inner.guard.__exit__(*args)
            def evidence(inner): return inner.guard.evidence()
            def check(inner):
                inner.guard.check(); return inner.budget.remaining_seconds(parent_api.tick_ms())
            def run_job(inner,job,protocol): return dispatch(job,protocol,inner.check())
        def dispatch(job,protocol,remaining):
            packet=Path(job["command"][-1]); p=json.loads(packet.read_text(encoding="utf-8"))
            state=json.loads((Path(p["output"])/"stage_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"],"RUNNING")
            jobs.append(dict(kind=p["kind"],remaining=remaining))
            if pre_fit_mutation and p["kind"]=="trajectory": pre_fit_mutation(p)
            if tamper:
                p["kind"]="replay-training"; packet.write_text(json.dumps(p),encoding="utf-8")
            outcome=final_status if p["kind"]=="final-verify" and final_status else status
            if outcome=="COMPLETE":
                child=api_factory(); child.current_pid=lambda:8
                handles=dict(read_handle=11,write_handle=12,parent_handle=10,nonce="3"*64)
                if "run_id" in p:
                    import msb_execution_safety as safety
                    binding=safety._binding_from_metadata(p,packet,handles,8,child)
                    child.inbox.append(safety.control_message(binding,"GRANT",lease_until_tick_ms=min(child.tick_ms()+30000,p["deadline_tick_ms"])))
                with patch.object(runner,"NativeMSBAPI",return_value=child,create=True):
                    if "handles" in inspect.signature(runner.worker).parameters: runner.worker(packet,handles=handles)
                    else: runner.worker(packet)
            return dict(job_id=job["job_id"],status=outcome,attempts=1,alive=False,
                        elapsed_seconds=.01,parent_binding_verified=True,job_assigned=True,
                        supervision=dict(power_request_released=outcome!="POWER_RELEASE_FAILED"))
        from contextlib import ExitStack
        def fit(*args,**kwargs):
            self.fit_calls.append(True)
            if not allow_fits: raise AssertionError("fit reached before preflight failure")
            return dict(scope="SYNTHETIC_ORCHESTRATION_ONLY",complete=True)
        def load(r,e,a):
            self.load_calls.append(e["fold"])
            if folds is None: raise AssertionError("real data opened")
            return folds[e["fold"]]
        with ExitStack() as stack:
            for name,value in (("WindowsAwakeGuard",guard_factory),("NativeEnvironmentAPI",lambda:None),("require_ac_power",lambda:None),
                               ("run_supervised_job",dispatch),("stage_elapsed_seconds",lambda:0.),("NativeMSBAPI",lambda:parent_api),
                               ("MSBParentSupervisor",Parent)):
                stack.enter_context(patch.object(runner,name,value,create=True))
            stack.enter_context(patch.object(runner.time,"monotonic",side_effect=monotonic))
            stack.enter_context(patch.object(runner,"load_prepared_fold",side_effect=load))
            if folds is not None:
                from msb_role_data import identity_digest
                protocol=dict(runner.synthetic_protocol(),authorization={},output_relative="stage",prepared_entries=[])
                for i,item in enumerate(folds):
                    protocol["prepared_entries"].append(dict(fold=i,relative_path=f"{runner.PREPARED_BASE}/prepare_fold{i}/result.pt",sha256="1"*64,
                        **{f"registered_{part}_ids_digest":identity_digest([r["image_id"] for r in item[part]["records"]]) for part in ("train","validation")}))
                if registered_path: protocol["prepared_entries"][0]["relative_path"]=registered_path
                stack.enter_context(patch.object(runner,"_read_protocol",return_value=protocol))
                stack.enter_context(patch.object(runner,"validate_protocol"))
                def registered(*args,**kwargs):
                    if registration_exhausted: clock[0]+=7200
                    return {}
                stack.enter_context(patch.object(runner,"_registered",side_effect=registered))
                stack.enter_context(patch.object(runner,"fit_trajectory",side_effect=fit))
                stack.enter_context(patch.object(runner,"fit_convex_fold",side_effect=fit))
                if allow_fits:
                    import analyze_msb_role_mechanism as analysis
                    stack.enter_context(patch.object(analysis,"collect_saved_predictions",side_effect=analysis_callback or
                        (lambda *a:dict(scope="SYNTHETIC_ORCHESTRATION_ONLY",delivery_verified=False))))
                    stack.enter_context(patch.object(analysis,"verify_saved_mechanism",return_value=dict(delivery_verified=True)))
                    def export(evidence,out):
                        if expire_export and out.name==expire_export: clock[0]+=7200
                        return evidence
                    stack.enter_context(patch.object(analysis,"write_mechanism_report",side_effect=export))
            state=runner.run_stage(self.root,self.root/"fixture.json" if folds is not None else None,self.root/"stage",
                                   "fit-mechanism" if folds is not None else "synthetic")
        return state,jobs

    def test_locally_valid_bad_threefold_preflight_has_zero_fit_calls(self):
        folds=make_prepared_fixture()["folds"]
        folds[1]=copy.deepcopy(folds[0]); folds[1]["inner_fold"]=1
        state,jobs=self.simulate(folds=folds)
        self.assertEqual(state["status"],"FAILED")
        self.assertEqual([j["kind"] for j in jobs],["load"]*3)
        self.assertEqual(self.fit_calls,[])
        self.assertEqual(state["actual_neural_fits"],0)
        self.assertEqual(state["actual_convex_fits"],0)

    def test_nonfit_training_record_fails_before_any_optimizer(self):
        folds=make_prepared_fixture()["folds"]
        folds[0]["train"]["records"][0]["tc_subset"]="other"
        state,jobs=self.simulate(folds=folds)
        self.assertEqual(state["status"],"FAILED")
        self.assertTrue(all(j["kind"]=="load" for j in jobs))
        self.assertEqual(self.fit_calls,[])

    def test_preflight_artifact_tamper_denied_before_fold_load(self):
        def tamper(packet):
            path=Path(packet["output"])/"runs/load_fold0/result.json"
            path.write_text(path.read_text()+"\n",encoding="utf-8")
        state,jobs=self.simulate(folds=make_prepared_fixture()["folds"],pre_fit_mutation=tamper)
        self.assertEqual(state["status"],"FAILED")
        self.assertIn("hash",state["error"])
        self.assertEqual(self.load_calls,[0,1,2])
        self.assertEqual(self.fit_calls,[])

    def test_forged_passed_state_without_live_parent_denied(self):
        packet=self.root/"packet.json"; packet.write_text('{"preflight_status":"PASSED"}',encoding="utf-8")
        with patch.object(runner,"load_prepared_fold",side_effect=AssertionError("unmanaged data access")):
            with self.assertRaises(ValueError): runner.worker(packet)
        self.assertFalse(packet.with_name("task_started.json").exists())

    def test_metadata_registration_time_is_not_refunded(self):
        state,jobs=self.simulate(folds=make_prepared_fixture()["folds"],registration_exhausted=True)
        self.assertEqual(state["status"],"TIMED_OUT")
        self.assertEqual(jobs,[])
        self.assertEqual(self.load_calls,[])

    def test_analysis_and_final_export_share_original_deadline(self):
        for target in ("analysis_summary","verified_summary"):
            with tempfile.TemporaryDirectory() as folder:
                previous=self.root; self.root=Path(folder)
                try: state,jobs=self.simulate(folds=make_prepared_fixture()["folds"],allow_fits=True,expire_export=target)
                finally: self.root=previous
            self.assertEqual(state["status"],"TIMED_OUT",state.get("error"))
            self.assertFalse(state["delivery_verified"])
            self.assertEqual(state["deadline_tick_ms"]-state["start_tick_ms"],7200000)

    def test_preflight_scope_does_not_allow_raw_or_stop_fallback(self):
        state,jobs=self.simulate(folds=make_prepared_fixture()["folds"],registered_path="projector_stop.pt")
        self.assertEqual(state["status"],"FAILED")
        self.assertEqual(self.fit_calls,[])
        self.assertTrue(all(j["kind"]=="load" for j in jobs))

    def test_draft_old_or_consumed_approval_opens_nothing(self):
        for status in ("DRAFT","FAILED","MECHANISM_COMPLETE","APPROVED"):
            with self.assertRaises(ValueError):
                runner.validate_protocol(self.root,dict(status=status),"fit-mechanism")
        with self.assertRaises(ValueError):
            runner.run_stage(self.root,None,self.root/"new","fit-mechanism")
        self.assertFalse((self.root/"new").exists())
        for key,value in (("fit_path","stop.pt"),("authorization",{"human_approved":True})):
            protocol=dict(runner.synthetic_protocol(),**{key:value})
            with self.assertRaises(ValueError): runner.validate_protocol(self.root,protocol,"synthetic")

    def test_running_published_before_dispatch(self):
        state,jobs=self.simulate()
        self.assertEqual(state["status"],"SYNTHETIC_COMPLETE")
        self.assertEqual(len(jobs),80)
        self.assertEqual(state["actual_neural_fits"],0)
        self.assertEqual(state["actual_convex_fits"],0)
        self.assertFalse(state["real_data_opened"])
        self.assertTrue(state["power"]["power_request_released"])

    def test_remaining_budget_reaches_final_verification(self):
        state,jobs=self.simulate()
        self.assertEqual(jobs[-1]["kind"],"final-verify")
        self.assertGreater(jobs[-1]["remaining"],0)
        self.assertLess(jobs[-1]["remaining"],jobs[0]["remaining"])
        self.assertEqual(state["state_history"][-1],"SYNTHETIC_COMPLETE")

    def test_timeout_power_release_interrupt_fail_closed(self):
        for status in ("TIMED_OUT","SUPERVISION_INTERRUPTED","POWER_RELEASE_FAILED"):
            with self.subTest(status=status):
                with tempfile.TemporaryDirectory() as directory:
                    previous=self.root; self.root=Path(directory)
                    state,jobs=self.simulate(status)
                    self.root=previous
                self.assertEqual(state["status"],status)
                self.assertEqual(len(jobs),1)
                self.assertTrue(state["power"]["power_request_released"])

    def test_final_verify_failure_is_terminal(self):
        state,jobs=self.simulate(final_status="TIMED_OUT")
        self.assertEqual(state["status"],"TIMED_OUT")
        self.assertEqual(len(jobs),80)
        self.assertFalse(state.get("delivery_verified",False))

    def test_existing_output_symlink_and_invalid_worker_rejected(self):
        existing=self.root/"old"; existing.mkdir()
        with self.assertRaises(FileExistsError): runner.run_stage(self.root,None,existing,"synthetic")
        with self.assertRaises(ValueError): runner.run_stage(self.root,None,self.root/".."/"escape","synthetic")
        with patch.object(Path,"is_symlink",return_value=True):
            with self.assertRaises(ValueError): runner.run_stage(self.root,None,self.root/"linked","synthetic")
        packet=self.root/"bad.json"
        packet.write_text(json.dumps(dict(kind="evaluate-stop")),encoding="utf-8")
        with self.assertRaises(ValueError): runner.worker(packet)

    def test_no_resume_stop_or_training_replay(self):
        state,jobs=self.simulate()
        with self.assertRaises(FileExistsError): runner.run_stage(self.root,None,self.root/"stage","synthetic")
        first=self.root/"stage"/"runs"/"load_fold0"/"packet.json"
        with self.assertRaises((ValueError,FileExistsError)): runner.worker(first)
        for kind in ("replay-training","prepare-raw","refit","pca","evaluate-stop"):
            packet=self.root/(kind+".json")
            packet.write_text(json.dumps(dict(kind=kind)),encoding="utf-8")
            with self.assertRaises(ValueError): runner.worker(packet)

    def test_packet_tamper_does_not_start_worker(self):
        state,jobs=self.simulate(tamper=True)
        self.assertEqual(state["status"],"SUPERVISION_INTERRUPTED")
        self.assertEqual(len(jobs),1)
        self.assertFalse((self.root/"stage"/"runs"/"load_fold0"/"task_started.json").exists())

    def test_protocol_scope_checked_before_file_read(self):
        with patch.object(Path,"read_text",side_effect=AssertionError("unscoped file opened")):
            with self.assertRaises(ValueError):
                runner.run_stage(self.root,self.root.parent/"outside.json",self.root/"new","fit-mechanism")

    def test_missing_parent_release_evidence_fails(self):
        class NoRelease(FakeGuard):
            def __exit__(self,*args): pass
        created=[]
        def factory(**kwargs):
            guard=NoRelease() if not created else FakeGuard()
            created.append(guard); return guard
        state,jobs=self.simulate(guard_factory=factory)
        self.assertEqual(len(jobs),80)
        self.assertEqual(state["status"],"POWER_RELEASE_FAILED")

    def test_cli_failure_does_not_pass_due_to_verified_flag(self):
        args=["run_msb_role_mechanism.py","--stage","synthetic","--output",str(self.root/"unused")]
        with patch.object(sys,"argv",args),patch.object(runner,"run_stage",return_value=dict(status="POWER_RELEASE_FAILED",delivery_verified=True)),patch.object(sys,"stdout",io.StringIO()):
            self.assertEqual(runner.main(),1)

    def test_real_path_worker_fixture_has_complete_status_and_release(self):
        state,jobs=self.simulate(folds=make_prepared_fixture()["folds"])
        result=json.loads((self.root/"stage/runs/load_fold0/result.json").read_text())
        self.assertEqual(result["status"],"COMPLETE")
        self.assertTrue(result["worker_power"]["power_request_released"])
        self.assertTrue(result["worker_execution"]["parent_binding_verified"])

    def real_metadata_fixture(self):
        protocol=dict(runner.synthetic_protocol(),status="APPROVED_REAL_ONCE",
                      authorization=dict(human_approved=True,scope="msb_rre_v1_fit_only_once",user_message_id="SYNTHETIC_METADATA_FIXTURE_NEVER_REGISTERED"),
                      source_sha256={},spec_sha256=runner.SPEC_SHA,repair_spec_sha256=runner.REPAIR_SPEC_SHA,
                      protocol_path="docs/experiment_protocols/msb_role_mechanism_v1.json",
                      output_relative="outputs/training/msb_role_mechanism_v1/development_fold_0",
                      environment=runner.runtime_environment(),prepared_entries=[])
        for name in runner.SOURCE_FILES:
            path=self.root/name; path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text("# synthetic identity fixture, never executed\n",encoding="utf-8")
            protocol["source_sha256"][name]=runner.file_digest(path)
        spec=self.root/runner.SPEC_PATH; spec.parent.mkdir(parents=True,exist_ok=True)
        spec.write_bytes((Path(__file__).resolve().parents[1]/runner.SPEC_PATH).read_bytes())
        repair=self.root/runner.REPAIR_SPEC_PATH; repair.parent.mkdir(parents=True,exist_ok=True)
        repair.write_bytes((Path(__file__).resolve().parents[1]/runner.REPAIR_SPEC_PATH).read_bytes())
        for fold in range(3):
            protocol["prepared_entries"].append(dict(fold=fold,relative_path=f"{runner.PREPARED_BASE}/prepare_fold{fold}/result.pt",sha256="1"*64,registered_train_ids_digest="2"*64,registered_validation_ids_digest="3"*64))
        return protocol

    def test_complete_metadata_and_source_identity_without_input_open(self):
        protocol=self.real_metadata_fixture()
        with patch.object(runner,"ROOT",self.root),patch.object(runner,"load_prepared_fold",side_effect=AssertionError("data read during metadata validation")):
            runner.validate_protocol(self.root,protocol,"fit-mechanism")
            name=next(iter(protocol["source_sha256"]))
            (self.root/name).write_text("# changed source fixture\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"source"):
                runner.validate_protocol(self.root,protocol,"fit-mechanism")
        self.assertFalse((self.root/"outputs").exists())

    def test_code_only_approval_cannot_be_relabelled_real(self):
        protocol=self.real_metadata_fixture()
        for token in ("call_978b15d460804fa2982cb750a1d4b4ad","call_1690e5bbad7a415f8880b0ff473b9521","call_3d401d8b809843e0b7103995a772caad"):
            protocol["authorization"]["user_message_id"]=token
            with patch.object(runner,"ROOT",self.root):
                with self.assertRaisesRegex(ValueError,"approval"):
                    runner.validate_protocol(self.root,protocol,"fit-mechanism")

    def test_analysis_scope_is_bound_to_existing_worker_guard(self):
        import analyze_msb_role_mechanism as analysis
        def collect(root,out,proto):
            context=analysis._CONTEXT.get()
            self.assertIsNotNone(context,"Worker did not bind the active supervised analysis scope")
            self.assertEqual(context["kind"],"analysis")
            self.assertIsInstance(context["guard"],runner.MSBWorkerScope)
            return dict(delivery_verified=False)
        state,jobs=self.simulate(folds=make_prepared_fixture()["folds"],allow_fits=True,analysis_callback=collect)
        self.assertEqual(state["status"],"MECHANISM_COMPLETE",state.get("error"))
        self.assertEqual(len(jobs),80)
        self.assertEqual(len(self.fit_calls),75)
        self.assertIsNone(analysis._CONTEXT.get())
