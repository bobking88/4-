# Role-Partitioned Granularity Gate (RPG-HRGV) Design

## Purpose

This document specifies an exploratory successor to CGDC-RSG-HRGV for the first paper. It addresses a structural property of the current labels: a 17-species posterior is mapped deterministically to four beneficiation-role classes. The proposed network must distinguish uncertainty *within* a role from uncertainty *between* roles before routing role evidence.

The task remains closed-set visual recognition on publicly available mineral specimen images. It does not estimate grade, recovery, industrial sorting performance, mineral chemistry, or operating-stage actions.

## Motivation and Novelty Boundary

Generic disagreement-aware calibration, generic multi-task learning, attention, Focal Loss, and multi-label mineral recognition are not claimed as innovations. In particular, CGDC's Jensen-Shannon-triggered residual calibration is retained only as a formal ablation because a 2026 cross-modal disagreement-calibration study is conceptually adjacent.

RPG-HRGV instead uses the fixed mineral-species-to-beneficiation-role partition as a model constraint. The paper may claim only a task-specific contribution: a role-partitioned uncertainty representation and gate for mineral species evidence under a fixed four-role decision protocol. It must not claim to be the first generic hierarchical or uncertainty-aware network.

## Labels and Notation

For image `x`, let `S` be a species label in the fixed species set `K`, and let `R=phi(S)` be a deterministic role in the four-role set `R`. The binary partition matrix `M` has `M[r,k]=1` iff `phi(k)=r`; every species belongs to exactly one role.

The backbone feature is `h=f_theta(x)`. The species expert produces `p_s(k|x)`, the direct role expert produces `p_d(r|x)`, and mapped role evidence is

`p_m(r|x) = sum_{k:phi(k)=r} p_s(k|x) = (M p_s)_r`.

## Role-Partitioned Uncertainty

The mapped role entropy is the between-role uncertainty

`U_between(x) = H[p_m(.|x)]`.

For each role with positive mapped mass, the conditional species posterior is

`p_s(k|r,x) = p_s(k|x)/p_m(r|x),  k in phi^{-1}(r)`.

The expected within-role species uncertainty is

`U_within(x) = sum_r p_m(r|x) H[p_s(.|r,x)]`.

By the chain rule for a deterministic partition,

`H[p_s(.|x)] = U_between(x) + U_within(x)`.

`U_between` represents uncertainty relevant to the four-role decision. `U_within` represents residual ambiguity among mineral species already assigned to the same role. The implementation must keep these two scalar features separate; it must not substitute total species entropy for role uncertainty.

## Network

RPG-HRGV retains the EfficientNet-B0 backbone, direct role head, species head, fixed role matrix, regret-supervised gate training, and residual hard-negative verifiers from RSG-HRGV. It removes CGDC's posterior residual calibration from the main path.

The RPG gate takes

`z_RPG = [h, H(p_d), U_between, U_within, |H(p_d)-U_between|]`.

It produces `g_RPG=sigmoid(MLP(z_RPG))` and fuses existing role evidence:

`p_f = g_RPG p_d + (1-g_RPG) p_m`.

The existing residual hard-negative verifier transforms `p_f` into the final role posterior. The gate remains trained with the registered detached, soft oracle regret target. Consequently, no true test label is present in inference.

## Objective

The initial objective intentionally reuses the verified RSG-HRGV loss:

`L_RPG = L_final + lambda_d L_direct + lambda_s L_species + lambda_c L_consistency + lambda_v L_verifier + lambda_h L_hard-negative + lambda_g L_gate-regret`.

No new regularizer is added until an ablation demonstrates that it supplies information beyond the role partition. This restriction prevents the first RPG experiment from conflating the gate's structural contribution with a large bundle of loss changes.

## Formula-Level Statements

### P-R1: Exact partitioned entropy identity

For the deterministic mapping `R=phi(S)`, `H(S|x)=H(R|x)+H(S|R,x)`. Under the definitions above, this is exactly `H[p_s]=U_between+U_within`.

### P-R2: Role sufficiency of mapped species evidence

For any two species posterior vectors `q_s` and `q'_s` with `M q_s=M q'_s`, the induced four-role posterior is identical. Therefore a redistribution of probability only among species inside the same role cannot, by itself, change a decision that depends only on `R`.

### P-R3: Fusion validity and evidence envelope

For `g_RPG` in `[0,1]`, every coordinate of `p_f` lies between the corresponding direct and mapped role probabilities. Also `p_f` lies on the four-role probability simplex. These are formula-level properties independent of data.

### P-R4: Gate information separation

The gate input exposes `U_between` and `U_within` as distinct coordinates. Thus total species uncertainty cannot be mathematically conflated with mapped role uncertainty by an implicit scalar substitution. This is an architectural separation statement, not a claim that the learned MLP will use every coordinate.

## Registered Experiments

The new paper experiment must use the existing fixed split and the three formal seeds `20260727`, `20260728`, and `20260729`. It must compare:

1. RSG-HRGV complete baseline.
2. RPG-HRGV complete.
3. RPG gate without `U_within`.
4. RPG gate without `U_between`.
5. RSG-HRGV plus total species entropy only.

All configurations use the same backbone, optimizer, image transformations, split groups, training epochs, and verifier policy. Evaluation includes Accuracy, Macro F1, target recall, target miss rate, Ti-bearing-negative-to-target intrusion, metallic-hard-negative-to-target intrusion, Brier score, ECE, gate-routing regret, and a paired seed-and-cluster Bootstrap against RSG-HRGV.

## Acceptance Criteria and Claim Boundary

P-R1 through P-R4 require unit tests. Empirical superiority is not assumed: the report may state it only if a paired confidence interval supports the relevant metric. If RPG does not improve the registered risk or calibration metrics, it remains a negative, well-controlled ablation and the first paper should keep RSG-HRGV as the main model.

The report must explicitly state that this architecture is evaluated only on the public-specimen four-role protocol. The stage-conditioned decision graph, reject-to-test policy, source-holdout generalization, and industrial deployment remain separate later research questions.
