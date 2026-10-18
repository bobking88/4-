"""Bounded native no-data supervision acceptance; no scientific input or model path."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

SOURCE_FILES = (
    "scripts/analyze_ic_role_mechanism.py",
    "scripts/analyze_lc_role_adaptation.py",
    "scripts/analyze_misclassifications.py",
    "scripts/analyze_msb_role_mechanism.py",
    "scripts/anchor_linear_controls.py",
    "scripts/audit_abmp_candidate_capacity.py",
    "scripts/check_lc_native_environment.py",
    "scripts/frozen_visual_probe.py",
    "scripts/generate_msb_role_figure.py",
    "scripts/hrgv_network.py",
    "scripts/ic_role_audit.py",
    "scripts/ic_role_data.py",
    "scripts/ic_role_model.py",
    "scripts/ic_role_training.py",
    "scripts/lc_execution_safety.py",
    "scripts/lc_role_adapter.py",
    "scripts/lc_role_audit.py",
    "scripts/lc_role_data.py",
    "scripts/lc_role_training.py",
    "scripts/mineral_hierarchy.py",
    "scripts/msb_execution_safety.py",
    "scripts/msb_role_audit.py",
    "scripts/msb_role_controls.py",
    "scripts/msb_role_data.py",
    "scripts/msb_role_model.py",
    "scripts/msb_role_training.py",
    "scripts/run_anchor_linear_controls.py",
    "scripts/run_frozen_visual_probe.py",
    "scripts/run_ic_role_mechanism.py",
    "scripts/run_lc_role_adaptation.py",
    "scripts/run_msb_role_mechanism.py",
    "scripts/run_seen_unseen_gate_study.py",
    "scripts/run_tc_oos_rsg_experiments.py",
    "scripts/run_verifier_strong_controls.py",
    "scripts/run_verifier_trust_development.py",
    "scripts/tc_oos_rsg.py",
    "scripts/train_hierarchical_mineral_classifier.py",
    "scripts/train_hrgv_mineral_classifier.py",
    "scripts/train_mineral_classifier.py",
    "scripts/train_role_aware_mineral_classifier.py",
    "scripts/verifier_risk.py",
    "scripts/verifier_strong_controls.py",
    "scripts/verify_verifier_trust_delivery.py",
    "scripts/check_msb_native_supervision.py",
    "tests/test_check_msb_native_supervision.py",
    "tests/run_msb_native_supervision_synthetic_acceptance.py",
)
CASES = (
    dict(case_id="N1", limit=90, seconds=45),
    dict(case_id="N2", limit=45, exit_code=23),
    dict(case_id="N3", limit=75, linger_seconds=35),
    dict(case_id="N4", limit=45, exit_code=31),
)
NON_NATIVE_IDS = {
    "call_a42aed1b7bab474bb17246c3df69dc4e", "call_6961d4dac6b64d7aba8e538f01c5e097",
    "call_5101ffc4ea2a44459043052c470036ce", "call_0e89f2cb80134c049e4ef3aaf697a7b0",
    "call_aa6d081524a94817b20cceb8fbd25053", "call_a4c9ef81bcea447aa7350f7cd9aff51c",
}
FIXED = dict(protocol="msb_native_supervision_v1", status="APPROVED_NATIVE_ONLY",
             scope="NATIVE_SUPERVISION_NO_DATA", execution_contract="msb_parent_bound_v2",
             output_relative="outputs/msb_native_supervision_v1/acceptance_once",
             max_seconds=480, case_order=["N1","N2","N3","N4"],
             case_limits=[90,45,75,45], require_ac=True, max_attempts=1,
             real_training_authorized=False)


def remaining_acceptance_seconds(start_monotonic: float, start_tick_ms: int, *,
                                 now_monotonic: float, now_tick_ms: int) -> float:
    if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0
            for v in (start_monotonic,now_monotonic))
            or type(start_tick_ms) is not int or type(now_tick_ms) is not int
            or start_tick_ms<0 or now_tick_ms<start_tick_ms or now_monotonic<start_monotonic):
        raise ValueError("Invalid or backwards acceptance clock")
    remaining=480.-max(now_monotonic-start_monotonic,(now_tick_ms-start_tick_ms)/1000.)
    if remaining<=0: raise RuntimeError("TIMED_OUT: cumulative native acceptance")
    return remaining


def validate_native_protocol(root: Path, protocol: dict, *, approval: dict) -> dict:
    if not isinstance(protocol,dict) or not isinstance(approval,dict):
        raise ValueError("Explicit native protocol and human approval required")
    if set(protocol)!=set(FIXED)|{"approval_call_id","approval_answer","source_sha256"}:
        raise ValueError("No extra input, experiment or execution fields")
    for key,value in FIXED.items():
        if type(protocol[key]) is not type(value) or protocol[key]!=value:
            raise ValueError("Fixed native acceptance field differs: "+key)
    if any(type(v) is not int for v in protocol["case_limits"]):
        raise ValueError("Integer fixed case limits required")
    token=protocol["approval_call_id"]
    if (not isinstance(token,str) or not re.fullmatch(r"call_[A-Za-z0-9]{16,64}",token)
            or token in NON_NATIVE_IDS or not isinstance(protocol["approval_answer"],str)
            or not protocol["approval_answer"].strip()
            or approval.get("call_id")!=token or approval.get("answer")!=protocol["approval_answer"]
            or approval.get("scope")!="NATIVE_SUPERVISION_NO_DATA"
            or approval.get("confirmed_from_human") is not True
            or not re.fullmatch(r"[a-f0-9]{40}",str(approval.get("registration_commit","")))):
        raise ValueError("New exact human native permission and registration required")
    sources=protocol["source_sha256"]
    if (not isinstance(sources,dict) or set(sources)!=set(SOURCE_FILES)
            or any(not isinstance(v,str) or not re.fullmatch(r"[a-f0-9]{64}",v) for v in sources.values())):
        raise ValueError("Exact native acceptance source closure required")
    destination=Path(root).resolve()/protocol["output_relative"]
    if destination.exists(): raise FileExistsError("Exclusive native output consumed; no retry")
    return copy.deepcopy(protocol)


def run_acceptance(root: Path, protocol_path: Path, *, execute_native: bool, approval: dict, services=None) -> dict:
    if execute_native is not True:
        raise ValueError("Explicit native execution permission required before any service or input")
    raise RuntimeError("Collector implementation not completed; native execution disabled")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--execute-native",action="store_true")
    parser.add_argument("--approval-call-id")
    parser.add_argument("--approval-answer")
    parser.add_argument("--registration-commit")
    args=parser.parse_args(argv)
    approval=dict(call_id=args.approval_call_id,answer=args.approval_answer,
                  registration_commit=args.registration_commit,
                  scope="NATIVE_SUPERVISION_NO_DATA",confirmed_from_human=True)
    result=run_acceptance(args.root,args.protocol,execute_native=args.execute_native,approval=approval)
    print(json.dumps(result),flush=True)
    return 0 if result.get("status")=="PASS_NATIVE_SUPERVISION_SCOPE" else 1


if __name__=="__main__":
    raise SystemExit(main())
