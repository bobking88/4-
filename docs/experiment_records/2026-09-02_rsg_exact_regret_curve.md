# RSG-HRGV Exact Regret Curve Evidence Record

## Purpose

This record documents the analytic figure used to explain the exact two-expert
convex-fusion routing-regret decomposition in the formal technical report and
the first-paper draft. It is a mechanism figure, not a new model-comparison
experiment.

## Figure Contract

- Core conclusion: under the implemented two-expert convex fusion, routing
  regret is exactly controlled by the product of gate deviation and normalized
  expert probability gap.
- Formula: let `M=max(a,b)`, `d=|a-b|`, `delta=|g-g_o|`, and
  `u=delta*d/M`. Then `p_g=M-delta*d` and `r=-log(1-u)`.
- Panel a: analytic curve of `r=-log(1-u)` and its first-order reference
  `r=u`.
- Panel b: analytic sensitivity curves for fixed `d/M` values of 0.15, 0.35,
  0.60, and 0.85 as gate deviation changes.
- Evidence note: high-precision checkpoint replay validates the implemented
  exact decomposition within a predefined numerical tolerance.

## Frozen Replay Evidence

Source summary:

`outputs/business_metrics/rsg_hrgv/gate_reliability/gate_reliability_summary.json`

| Quantity | Value |
|---|---:|
| Checkpoint replays | 9 |
| Image evaluations | 10,242 |
| Maximum exact-decomposition absolute residual | 8.44e-07 |
| Numerical tolerance | 2.00e-06 |
| Exact-decomposition violations | 0 |

The finite residual is consistent with float32 probability and loss values
being exported and re-evaluated in finite precision. The record therefore
states tolerance-level numerical consistency, not mathematical zero error.

## Outputs

The figure generator is
`scripts/generate_rsg_exact_regret_figure.py`. It writes the following bundle
under `outputs/paper_figures_v3/`:

- `fig_rsg_exact_regret_decomposition.png`
- `fig_rsg_exact_regret_decomposition.svg`
- `fig_rsg_exact_regret_decomposition.pdf`
- `fig_rsg_exact_regret_decomposition.tiff`
- `fig_rsg_exact_regret_decomposition_source.json`

The SVG and PDF retain editable text. The TIFF is exported at 600 dpi; the PNG
is the visual-review preview. The source JSON preserves the evidence summary
and claim boundary used in the figure note.

## Claim Boundary

The analytic curves explain a pre-defined routing-risk quantity. They do not
provide a new comparison of Accuracy, Macro F1, industrial sorting, grade,
recovery, elemental content, independent external generalization, or OOD
recognition. The visual result should be cited together with the high-precision
replay, not presented as stand-alone performance evidence.
