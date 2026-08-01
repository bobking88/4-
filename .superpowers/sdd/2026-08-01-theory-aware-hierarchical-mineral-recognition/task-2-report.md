# Task 2 Implementation and Self-Review Report

## Scope delivered

Implemented confidence-based selective-recognition risk analysis in `scripts/analyze_selective_recognition.py` and its unit tests in `tests/test_analyze_selective_recognition.py`.

The analysis consumes the three required full-hierarchical prediction files only:

- `formal_hierarchical_efficientnet_b0_seed20260727/test_predictions.csv`
- `formal_hierarchical_efficientnet_b0_seed20260728/test_predictions.csv`
- `formal_hierarchical_efficientnet_b0_seed20260729/test_predictions.csv`

It writes the requested JSON summary, Markdown summary, and Figure 9 PNG.

## Implementation details

- `calculate_selective_metrics(rows, thresholds)` filters retained records where `confidence >= threshold`.
- For every retained set it records coverage, overall retained risk, target-proxy miss rate, titanium-interference intrusion rate, and metallic-hard-negative intrusion rate.
- A rate with no applicable retained denominator is represented as `None`; it is never replaced by zero.
- The analysis uses thresholds from 0.00 through 0.95 in 0.05 increments.
- The JSON preserves each seed's values at every threshold and provides defined-value mean plus sample standard deviation.
- The Markdown and figure state that lower coverage means more samples are deferred for later inspection. They do not describe deferral as a decision, an industrial cost-optimal policy, or real XRF validation.
- The source resolver requires exactly the three named seed directories, preventing an accidental inclusion of another prediction family.
- `fixed_split_consumed_unchanged` is recorded as `true`; no split construction or modification occurs.

## Test-driven development record

1. Added the threshold-deferral test before the implementation module existed.
2. Ran `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_analyze_selective_recognition -v` and observed the expected `ModuleNotFoundError` for `analyze_selective_recognition`.
3. Implemented the metrics function and aggregation.
4. Added coverage for retained target-proxy and interference rates, undefined denominators, and seed-level aggregation.
5. Re-ran the focused test suite successfully: 4 tests passed.

## Final verification

The following commands completed successfully using the required training environment:

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_analyze_selective_recognition -v
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe scripts\analyze_selective_recognition.py --input-glob outputs\training\formal_hierarchical_efficientnet_b0_seed*\test_predictions.csv --output-dir outputs\theory_validation\selective_recognition --figure outputs\paper_figures_v1\fig9_selective_recognition.png
```

The real analysis confirmed three inputs and 20 recorded thresholds. The Figure 9 image was visually inspected: both requested Chinese panels are present and legible.

## Self-review

- Scope: only Task 2 analysis code, its test, requested deliverables, and this required task report are included. The pre-existing modified implementation-plan file is not included.
- Numerical handling: per-seed values remain intact; mean and sample standard deviation are calculated only from defined values. An undefined rate remains undefined in both seed-level and aggregate output.
- Interpretation: the output limits its claims to the frozen public-image test split. It makes no claim of industrial economics, process performance, recovery, grade, or physical XRF validation.
- Residual limitation: confidence thresholds are descriptive on this fixed evaluation split and have not been calibrated or externally validated; deferred records are recommendations for later inspection only.

## Fix Round 1/5: Exact Input Cardinality

### Finding

The input resolver previously collapsed glob matches into a dictionary keyed by parent seed-directory name. Multiple matching CSV files under an expected seed directory could therefore be overwritten silently, leaving one arbitrary path accepted instead of enforcing exactly one prediction file per required seed.

### Changed files

- `scripts/analyze_selective_recognition.py`: `_resolve_input_paths` now builds a list of matching paths for every expected seed and raises `ValueError` when any expected seed has zero or more than one match. The error includes missing, duplicate, and unexpected seed names.
- `tests/test_analyze_selective_recognition.py`: added a filesystem-backed regression test that creates one expected CSV for all three seeds plus an alternate CSV for one expected seed, then asserts that the resolver rejects the input.
- `.superpowers/sdd/2026-08-01-theory-aware-hierarchical-mineral-recognition/task-2-report.md`: appended this fix record.

### Test command and output

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_analyze_selective_recognition -v
```

Output: `Ran 5 tests in 0.521s` and `OK`. The new duplicate-file resolver test passed.
