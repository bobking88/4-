"""Execution gates tested with mocked process, power and clock APIs only."""
import json
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

    def simulate(self,status="COMPLETE",*,final_status=None,tamper=False,guard_factory=FakeGuard):
        jobs=[]; clock=[0.]
        def monotonic(): clock[0]+=.2; return clock[0]
        def dispatch(job,protocol,remaining):
            packet=Path(job["command"][-1]); p=json.loads(packet.read_text(encoding="utf-8"))
            state=json.loads((Path(p["output"])/"stage_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"],"RUNNING")
            jobs.append(dict(kind=p["kind"],remaining=remaining))
            if tamper:
                p["kind"]="replay-training"; packet.write_text(json.dumps(p),encoding="utf-8")
            outcome=final_status if p["kind"]=="final-verify" and final_status else status
            if outcome=="COMPLETE": runner.worker(packet)
            return dict(job_id=job["job_id"],status=outcome,attempts=1,alive=False,
                        elapsed_seconds=.01,supervision=dict(power_request_released=outcome!="POWER_RELEASE_FAILED"))
        with patch.object(runner,"WindowsAwakeGuard",guard_factory),patch.object(runner,"NativeEnvironmentAPI"),patch.object(runner,"require_ac_power"),patch.object(runner,"run_supervised_job",side_effect=dispatch),patch.object(runner.time,"monotonic",side_effect=monotonic),patch.object(runner,"stage_elapsed_seconds",return_value=0.),patch.object(runner,"load_prepared_fold",side_effect=AssertionError("real data opened")):
            state=runner.run_stage(self.root,None,self.root/"stage","synthetic")
        return state,jobs

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
        self.assertEqual(state["status"],"FAILED")
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
        output=self.root/"fixture"; folder=output/"runs"/"load_fold0"; folder.mkdir(parents=True)
        path=self.root/"docs"/"experiment_protocols"/"msb_role_mechanism_v1.json"
        path.parent.mkdir(parents=True)
        protocol=dict(authorization={},prepared_entries=[{}])
        path.write_text(json.dumps(protocol),encoding="utf-8")
        packet=dict(kind="load",fold=0,job_id="load_fold0",root=str(self.root),output=str(output),
                    stage="fit-mechanism",protocol_path=str(path),protocol_digest=runner.protocol_digest(protocol))
        packet_path=folder/"packet.json"; packet_path.write_text(json.dumps(packet),encoding="utf-8")
        state=dict(status="RUNNING",stage="fit-mechanism",active_job="load_fold0",protocol_digest=runner.protocol_digest(protocol),
                   packets={"load_fold0":runner.file_digest(packet_path)})
        (output/"stage_state.json").write_text(json.dumps(state),encoding="utf-8")
        with patch.object(runner,"validate_protocol"),patch.object(runner,"load_prepared_fold",return_value={}),patch.object(runner,"WindowsAwakeGuard",FakeGuard),patch.object(runner,"NativeEnvironmentAPI"):
            result=runner.worker(packet_path)
        self.assertEqual(result["status"],"COMPLETE")
        self.assertTrue(result["worker_power"]["power_request_released"])

    def real_metadata_fixture(self):
        protocol=dict(runner.synthetic_protocol(),status="APPROVED_REAL_ONCE",
                      authorization=dict(human_approved=True,scope="msb_rre_v1_fit_only_once",user_message_id="SYNTHETIC_METADATA_FIXTURE_NEVER_REGISTERED"),
                      source_sha256={},spec_sha256=runner.SPEC_SHA,
                      protocol_path="docs/experiment_protocols/msb_role_mechanism_v1.json",
                      output_relative="outputs/training/msb_role_mechanism_v1/development_fold_0",
                      environment=runner.runtime_environment(),prepared_entries=[])
        for name in runner.SOURCE_FILES:
            path=self.root/name; path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text("# synthetic identity fixture, never executed\n",encoding="utf-8")
            protocol["source_sha256"][name]=runner.file_digest(path)
        spec=self.root/runner.SPEC_PATH; spec.parent.mkdir(parents=True,exist_ok=True)
        spec.write_bytes((Path(__file__).resolve().parents[1]/runner.SPEC_PATH).read_bytes())
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
        protocol["authorization"]["user_message_id"]="call_978b15d460804fa2982cb750a1d4b4ad"
        with patch.object(runner,"ROOT",self.root):
            with self.assertRaisesRegex(ValueError,"approval"):
                runner.validate_protocol(self.root,protocol,"fit-mechanism")

    def test_analysis_scope_is_bound_to_existing_worker_guard(self):
        import analyze_msb_role_mechanism as analysis
        output=self.root/"fixture"; folder=output/"runs"/"analysis"; folder.mkdir(parents=True)
        path=self.root/"docs"/"experiment_protocols"/"msb_role_mechanism_v1.json"
        path.parent.mkdir(parents=True); protocol={}
        path.write_text(json.dumps(protocol),encoding="utf-8")
        packet=dict(kind="analysis",job_id="analysis",root=str(self.root),output=str(output),
                    stage="fit-mechanism",protocol_path=str(path),protocol_digest=runner.protocol_digest(protocol))
        packet_path=folder/"packet.json"; packet_path.write_text(json.dumps(packet),encoding="utf-8")
        state=dict(status="RUNNING",stage="fit-mechanism",active_job="analysis",protocol_digest=runner.protocol_digest(protocol),
                   packets={"analysis":runner.file_digest(packet_path)})
        (output/"stage_state.json").write_text(json.dumps(state),encoding="utf-8")
        def collect(root,out,proto):
            context=analysis._CONTEXT.get()
            self.assertIsNotNone(context,"Worker did not bind the active supervised analysis scope")
            self.assertEqual(context["kind"],"analysis")
            return dict(delivery_verified=False)
        with patch.object(runner,"validate_protocol"),patch.object(runner,"WindowsAwakeGuard",FakeGuard),patch.object(runner,"NativeEnvironmentAPI"),patch.object(analysis,"collect_saved_predictions",side_effect=collect),patch.object(analysis,"write_mechanism_report",side_effect=lambda e,o:e):
            result=runner.worker(packet_path)
        self.assertEqual(result["status"],"COMPLETE")
        self.assertIsNone(analysis._CONTEXT.get())
