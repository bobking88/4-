from __future__ import annotations

import html
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class TCOOSRSGFigureTests(unittest.TestCase):
    def test_figure_exports_architecture_bundle_and_theorem_evidence(self) -> None:
        from generate_tc_oos_rsg_figure import generate_tc_oos_rsg_figure

        invariants = {
            "row_count": 12,
            "simplex_violation_count": 0,
            "target_safety_violation_count": 0,
            "decomposition_violation_count": 0,
            "convex_nll_bound_violation_count": 0,
            "nonfinite_or_negative_count": 0,
            "cross_fold_group_overlap_count": 0,
            "max_target_harm": 0.0,
            "max_decomposition_residual": 1e-9,
            "tolerance": 1e-6,
            "all_invariants_pass": True,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prefix = root / "fig_tc_oos_rsg_architecture"
            invariant_path = root / "tc_oos_rsg_invariants.json"

            source = generate_tc_oos_rsg_figure(
                prefix, invariants=invariants, invariant_output=invariant_path
            )

            for suffix in (".png", ".pdf", ".svg"):
                output = prefix.with_suffix(suffix)
                self.assertTrue(output.exists(), suffix)
                self.assertGreater(output.stat().st_size, 1000, suffix)
            svg_lines = prefix.with_suffix(".svg").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertFalse(any(line != line.rstrip() for line in svg_lines))
            source_path = prefix.with_name(prefix.name + "_source.json")
            self.assertTrue(source_path.exists())
            self.assertTrue(invariant_path.exists())
            self.assertEqual(source["evidence_vector"]["dimension_count"], 18)
            self.assertEqual(
                len(source["evidence_vector"]["ordered_dimensions"]), 18
            )
            self.assertEqual(
                json.loads(invariant_path.read_text(encoding="utf-8")), invariants
            )

    def test_svg_contains_implemented_tensor_and_theorem_labels(self) -> None:
        from generate_tc_oos_rsg_figure import generate_tc_oos_rsg_figure

        invariants = {
            "scope": "one-fold smoke",
            "row_count": 4,
            "simplex_violation_count": 0,
            "target_safety_violation_count": 0,
            "decomposition_violation_count": 0,
            "convex_nll_bound_violation_count": 0,
            "nonfinite_or_negative_count": 0,
            "cross_fold_group_overlap_count": 0,
            "max_target_harm": 0.01,
            "max_decomposition_residual": 0.0,
            "tolerance": 1e-6,
            "all_invariants_pass": True,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prefix = root / "architecture"
            generate_tc_oos_rsg_figure(
                prefix,
                invariants=invariants,
                invariant_output=root / "invariants.json",
            )
            svg = html.unescape(
                prefix.with_suffix(".svg").read_text(encoding="utf-8")
            )

        for label in (
            "q0",
            "q_phi",
            "rho_tilde",
            "epsilon_T",
            "verifier-complete",
            "q_TC,T >= q0,T - epsilon_T",
            "explicit q0 fallback",
            "one-fold smoke",
        ):
            self.assertIn(label, svg)


if __name__ == "__main__":
    unittest.main()
