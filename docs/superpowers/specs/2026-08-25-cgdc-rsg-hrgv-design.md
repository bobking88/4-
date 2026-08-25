# CGDC-RSG-HRGV-Net Design Specification

## Purpose

This design upgrades RSG-HRGV-Net from two linear prediction heads on one shared EfficientNet-B0 representation into a cross-granularity expert architecture. The proposed **Cross-Granularity Evidence Decomposition and Disagreement Calibration** (CGDC) module:

1. creates separately parameterized role- and species-oriented feature paths after the shared backbone; and
2. corrects the RSG-fused role posterior only in proportion to observable disagreement between the two hierarchical experts.

The scope remains public mineral-specimen visual recognition. The model does not predict V/Ti grade, recovery, industrial sorting performance, or unknown-mineral/OOD performance.

## Novelty boundary

Current RSG-HRGV uses one feature vector \(h\) and two linear classification heads. The direct-role and species-mapped experts share all visual representation before their final linear layers, which limits their complementarity.

Supervised routing, hierarchical gating and gradient-decoupled MoE are established research topics. The paper will not claim those generic concepts as new. The proposed contribution is narrower:

> Given a fixed mineral-species-to-beneficiation-role mapping, construct decomposed cross-granularity evidence and a disagreement-triggered bounded posterior residual. The residual is inactive under hierarchical agreement and its induced log-odds change is quantitatively bounded by expert disagreement.

The working name is **CGDC-RSG-HRGV-Net** and must remain qualified as a proposed domain-specific architecture until final literature verification.

## Architecture

Let the EfficientNet-B0 backbone yield \(h=f_\theta(x)\in\mathbb R^{1280}\).

### Decomposed expert adapters

Two independent residual bottleneck adapters generate task-specific features:

\[
u_d=h+A_d(h),\qquad u_s=h+A_s(h).
\]

The direct role expert and species expert are

\[
p_d=\operatorname{softmax}(W_du_d),\qquad
p_s=\operatorname{softmax}(W_su_s),\qquad
p_m=A p_s,
\]

where \(A\in\{0,1\}^{4\times17}\) is the frozen species-to-role mapping. The mapped posterior is normalized after multiplication.

The optional decomposition regularizer is

\[
\mathcal L_{dec}=\frac1N\sum_i
\left(
\frac{A_d(h_i)^\top A_s(h_i)}
{\|A_d(h_i)\|_2\|A_s(h_i)\|_2+\epsilon}
\right)^2.
\]

Its pre-registered default coefficient is \(\lambda_{dec}=0.02\), without test-set tuning.

### Existing RSG fusion

The existing RSG gate is retained:

\[
p_f=g p_d+(1-g)p_m.
\]

The regret target and stop-gradient boundary remain unchanged. This separates new representation/calibration effects from the already registered RSG routing objective.

### Disagreement-triggered bounded calibration

Define the nonnegative disagreement gain

\[
\rho=1-\exp[-D_{JS}(p_d\Vert p_m)].
\]

Build a conflict feature from decomposed paths and posterior conflict:

\[
z_c=[u_d,u_s,u_d\odot u_s,|u_d-u_s|,
\log(p_d+\epsilon)-\log(p_m+\epsilon),
\bar H(p_d),\bar H(p_m),D_{JS}(p_d\Vert p_m)].
\]

The calibration MLP emits bounded residual logits

\[
s=\tanh\{f_c(z_c)\}\in[-1,1]^4,
\]

and the calibrated posterior is

\[
p_c=\operatorname{softmax}\left(\log(p_f+\epsilon)+\rho s\right).
\]

The existing target-versus-Ti and target-versus-metallic residual verifiers operate on \(p_c\), not \(p_f\), to yield final posterior \(q\).

### Objective

\[
\mathcal L_{CGDC}=\mathcal L_{RSG-HRGV}
+\lambda_{dec}\mathcal L_{dec}
+\lambda_{cal}\mathcal L_{cal},\qquad
\mathcal L_{cal}=-\frac1N\sum_i\log p_c(y_i\mid x_i).
\]

The pre-registered default is \(\lambda_{cal}=0.25\). The final post-verifier role loss remains the principal classification loss; \(\mathcal L_{cal}\) directly supervises the pre-verifier calibrated posterior.

## Theoretical properties

### Proposition 1: agreement identity

If \(p_d=p_m\), then \(D_{JS}(p_d\Vert p_m)=0\), \(\rho=0\), and

\[
p_c=p_f=p_d=p_m.
\]

The calibrator cannot alter a hierarchically agreeing prediction, regardless of its MLP parameters.

### Proposition 2: probability preservation

For every image, \(p_c\) belongs to the four-role probability simplex: \(p_{c,j}>0\) and \(\sum_jp_{c,j}=1\).

### Proposition 3: disagreement-adaptive log-odds bound

For any roles \(j,k\), because \(s_j,s_k\in[-1,1]\),

\[
\left|
\log\frac{p_{c,j}}{p_{c,k}}
-\log\frac{p_{f,j}}{p_{f,k}}
\right|
=\rho|s_j-s_k|
\le 2\rho.
\]

Correction is zero under agreement and bounded by observed expert disagreement. This is a network-level stability property, not an assertion of universal metric improvement.

### Proposition 4: RSG routing compatibility

RSG fusion \(p_f\) and its regret target are evaluated before CGDC calibration. The existing routing-regret theorem therefore remains applicable to expert fusion; CGDC is evaluated separately by final-role risk and calibration metrics.

## Controlled experiment

Each formal configuration uses the frozen manifest, EfficientNet-B0, coupled residual verifiers, formal seeds 20260727, 20260728 and 20260729, identical augmentation, optimizer, early stopping, and group-cluster paired Bootstrap.

| Configuration | Purpose |
|---|---|
| rsg_complete | Existing formal reference |
| cgdc_complete | Full adapters, disagreement gain, bounded residual and both new losses |
| cgdc_shared_features | Retains calibration but removes adapter decomposition |
| cgdc_unconditional | Sets \(\rho=1\), testing disagreement-triggering |
| cgdc_no_decomposition_loss | Sets \(\lambda_{dec}=0\), testing soft feature separation |

Primary metrics: Accuracy, Macro F1, target recall, target miss rate, Ti-to-target intrusion and metallic-to-target intrusion. Mechanism metrics: pre/post-calibration prediction changes, mean \(\rho\), ECE, Brier score and paired group-cluster Bootstrap intervals. Existing routing metrics are retained but are not CGDC's primary endpoint because they are defined at \(p_f\).

## Boundaries and deliverables

- New flags default to off, preserving the current RSG-HRGV behavior.
- The residual is four-role, bounded by tanh, and does not replace target-only verifiers.
- Inference has no access to ground-truth labels, mining stage or cost metadata.
- Unit tests cover each proposition, output dimensions, gradient paths and disabled-CGDC baseline equivalence.
- Deliverables include source code, architecture figure, theorem appendix, formal three-seed outputs, paired analysis, technical-report update and paper-method/results update.
