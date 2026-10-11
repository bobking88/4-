# LC-RFA-B v2 Complete Development Rerun: Verified Negative Result

## Execution and Verification

The human message `完整开发重跑` authorized the bounded v2 rerun registered at
`6e7af63932ab1fada75fd6e1200939819c874fd2`. The scientific design was unchanged.
One fit and one locked stop evaluation completed with exit 0. No retry, resume,
new images/groups, training replay or formal-report edit occurred.

- Actual fits: 392, plus four temperature fits. Exact S1 CV/final trajectories
  were reused under the registered rules, not copied from the failed v1 batch.
- Fit supervision: 396 workers including four preparations, all complete,
  exited/reaped and child system-awake requests released.
- Final locked groups: 26; T0 additionally evaluated as the anchor.
- Fit/evaluation consumed time: 2572.328 seconds, below 21600 seconds.
- One evaluation worker completed. All three seeds and all controls retained.
- Default verification: `VERIFIED_INFERENCE_ONLY`, 2124 OOF candidates and
  26 final groups replayed with exact tensor equality; no training replay.
- Histories and gradients have hash-integrity evidence, not an independently
  replayed optimizer trajectory. The verification is not full training reproduction.
- All 1776 historical/protected files, frozen sources, both formal-report
  copies and system power settings remained unchanged.
- A post-run process query found no `run_lc_role_adaptation.py` processes.

Host invocation/exit timestamps remain in the raw JSON records. The fit began
on 2026-10-11; final audit/archiving was performed on 2026-10-12. These separate
timestamps must not be presented as one uninterrupted measurement interval.

## Primary Development Outcome

**The registered development screen failed. No role-routing or bounded
net-benefit claim is supported under this frozen setup.**

H0, S0, R0 and U1 selected the 400-update boundary and were marked
`OPTIMIZATION_TRUNCATED` for all seeds. These twelve entries invalidate the
full screen; the screen therefore does not execute its per-seed or U1-dominance
checks. Its empty dominance list is not evidence of non-dominance.

There is also a direct negative observation against the non-truncated S1
control: R1 and S1 give identical argmax predictions on all 340 stop images
for every seed, with NLL differences below 1.7e-7. S1 and R1 both selected
regularization .001, alpha/beta maxima .5/.5 and 100 updates. The matched-budget
S1 selection is exactly the same S1 trajectory.

| Seed | R1 NLL | S1 NLL | R1 minus S1 | R1 correct targets | Net vs T0 | Ti false targets | Metallic false targets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261002 | .828850318 | .828850322 | -3.662e-9 | 40 | -3 | 10 | 6 |
| 20261003 | .829456226 | .829456057 | +1.691e-7 | 41 | -2 | 9 | 6 |
| 20261004 | .828356246 | .828356205 | +4.145e-8 | 40 | -3 | 10 | 6 |

T0 has NLL .835190593, 43 correct targets out of 66, 13 Ti false targets
out of 130 and 6 metallic false targets out of 47. R1 improves NLL over T0,
but loses correct targets and provides no material advantage over S1. This
does not meet the specified target-net-correction or all-seed matched-S1 criteria,
even without relaxing the separate truncated-control rejection.

| Model | Stop Accuracy | Stop Macro F1 | Target Recall |
| --- | ---: | ---: | ---: |
| T0 | 68.529% | 65.429% | 65.152% |
| S1, mean +/- sample SD | 68.725% +/- .340% | 65.158% +/- .377% | 61.111% +/- .875% |
| R1, mean +/- sample SD | 68.725% +/- .340% | 65.158% +/- .377% | 61.111% +/- .875% |

These are 340-image, repeatedly exposed development-stop observations, not
the historical 1284-image test results, independent validation or significance.
Three seeds are algorithm repeats on one data set, not independent samples.
All fit/stop groups, metrics and raw denominators are in `metrics_summary.csv`.

## Theory and Mechanism Evidence

All stored fit/stop numerical audits report zero violations. This supports
implementation conformity with the registered A/B/C inequalities, not novelty,
generalization, improved true-label risk or guaranteed target recall.

The same-weights main-path certificate remains

$$
\eta(x)=(\bar\alpha+\bar\beta)\varepsilon_f L_F b(x)^2,
\qquad b(x)=4t(x)(1-t(x)).
$$

Its protected-winner coverage for R1 on stop is 233/340, 230/340 and 238/340.
This compares adaptation against the same trained readout with adaptation
disabled, not against T0 or an independently trained ordinary model. Its
certificate is not applicable to unscaled auxiliary views.

R1 branch outputs are nonzero on all rows, but their maximum norm is only
about 3.28e-4, 5.26e-4 and 3.81e-4 across the respective seeds, well below the
unit norm cap. Each fixed-state leave-one-branch-out diagnostic changes zero
argmax predictions. Maximum R1/S1 log-probability differences are approximately
2.35e-5, 7.76e-5 and 2.99e-5. The branch paths are therefore numerically active
but have little observed decision effect in this run.

These observations motivate examining the existing initialization, gradient,
regularization and shared-readout traces before proposing another architecture.
They do not establish a causal diagnosis, prove role adaptation universally
ineffective, or authorize a larger grid, altered loss or another real rerun.

## Complete Evidence and Boundaries

Output: `outputs/training/lc_role_adaptation_v2/development_fold_0`.
`text_evidence_complete.zip` preserves all 1232 text artifacts, including
training histories, gradients, logs, fit locks and every per-image prediction
CSV. Every member was CRC- and SHA-256-checked against its original file.
`artifact_manifest.json` records all output-file identities before manifest/zip
creation; 797 tensor/packet files remain local with hashes, not in the archive.
Original mineral images, frozen cache and model weights are not distributed.

| Artifact | SHA-256 |
| --- | --- |
| Selection lock | `0ce4a702075c5b0b0fc437d54cb4adf6d4e654bad8f491121eaba0e93660a7ea` |
| Registered protocol | `2f716188280105361335129c0a2c4c56455755a415b21fe20a84a4ecc3b05555` |
| Analysis summary | `ef1d81f7c039120b3155ed2b473b929ecca567ff24d5045e86b7d81fcf091d24` |
| Default verification | `d15467d6c540714d71d3f7a716ebfdfd354cc79a78538d7a2404a8e35f8e9e3b` |
| Complete text archive, 13366107 bytes | `0291368f1480a5aa023bda032930b7ccddd11d8a423670c43e9083fd73ae53b9` |
| Formal report, unchanged | `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c` |

Supporting invocation, exit, fit-lock and post-run audits are under
`outputs/lc_execution_safety_qa/real_v2_*`. This authorization is consumed.
The next sensible work is a written, trace-based mechanism diagnosis and a
revised theoretical hypothesis, followed by separately approved experiments
only if warranted. No stable network benefit should enter the formal report
from this batch. The existing analytical structure may be described separately
with its assumptions and negative experimental findings.
