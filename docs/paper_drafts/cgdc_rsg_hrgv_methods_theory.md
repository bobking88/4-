# RSG-HRGV-Net with M-RPG Extension: Methods and Theoretical Statements

## Intended Paper Positioning

This paper studies closed-set visual recognition of vanadium-titanium-magnetite-related minerals from publicly available mineral specimen images. It does not infer V, Ti, or Fe grade, recovery, industrial sorting performance, or mineral chemistry from RGB images.

The primary network is RSG-HRGV: a cross-granularity direct-role/species-mapped-role fusion network with risk-supervised gating and residual verifiers. M-RPG is a theory-driven extension for a fixed four-role task. It separates species uncertainty into between-role and within-role components, normalizes each term by its available support, and constrains direct-role evidence allocation to be nondecreasing in role-relevant uncertainty. Both are evaluated using the same data split, ImageNet-pretrained EfficientNet-B0 backbone, and three registered random seeds.

### Related-work boundary

CGDC must not be presented as a new generic disagreement-calibration mechanism. HiRoC (2026, doi:10.1007/s44443-026-01163-x) also uses Jensen-Shannon disagreement, a bounded gate, and residual decision-space correction, albeit for multimodal conversational emotion recognition rather than a deterministic mineral-species-to-role hierarchy. The CGDC experiment is therefore retained as a controlled reliability ablation only. Original RPG-HRGV is retained as the decomposition baseline. M-RPG-HRGV is reported as a theory-driven target-recall extension, not as a replacement primary network: its overall Macro F1 and direct-ablation intervals do not establish a stable general gain under the present protocol.

## 1. Problem and Label Granularity

Each image x_i has a mineral species label s_i in a fixed 17-species vocabulary and a beneficiation-role label r_i in four roles: target_mineral, ti_bearing_negative, gangue_negative, and metallic_hard_negative. A fixed matrix M maps species probabilities to role probabilities. The proposed task preserves the species label as auxiliary evidence but evaluates the four-role posterior.

## 2. Cross-Granularity Evidence Decomposition

Let h=f_theta(x) denote the backbone feature. Direct role evidence and species evidence are separated by two residual adapters:

u_d = h + A_d(h),     u_s = h + A_s(h).

The direct role expert and species expert are

p_d = softmax(W_d u_d),     p_s = softmax(W_s u_s),     p_m = M p_s.

To discourage the two adapters from simply duplicating the same residual evidence, the model uses

L_dec = mean[cos^2(A_d(h), A_s(h))].

This term does not assert statistical independence. It is a trainable representation-level preference for non-collinear residual corrections under a shared backbone.

## 3. Regret-Supervised Evidence Fusion

The existing RSG fusion provides

p_f = g p_d + (1-g) p_m,     0 <= g <= 1.

The gate is trained with a soft oracle target derived from the true-class evidence gap while its regret branch is locally detached from backbone and expert parameters. This preserves the previously established routing analysis on the pre-calibration posterior p_f.

## 4. Role-Partitioned Granularity Gate (Baseline Decomposition)

The fixed role mapping M partitions the species posterior into four disjoint role groups. Let R be the role random variable induced by the species posterior p_s. The mapped posterior is p_m=M p_s. Its Shannon uncertainty satisfies the exact chain rule

H(S) = H(R) + H(S | R).

We denote H(R) by U_between and H(S | R) by U_within. U_between measures uncertainty across the four beneficiation-role labels. U_within measures unresolved species variation after the role group has been fixed. They are not interchangeable: redistributing probability among species inside a fixed role leaves p_m and U_between unchanged but changes U_within.

The RPG gate receives a shared visual feature and separated uncertainty signals:

z_RPG = [h, H(p_d), U_between, U_within, |H(p_d)-U_between|],

g = sigmoid(G(z_RPG)),

p_f = g p_d + (1-g) p_m.

The output remains a convex mixture of two four-role posteriors. RPG does not assume that a larger or smaller entropy is always preferable; it supplies the gate with the two granularities separately and tests their contribution with registered ablations.

### Proposition R1: Exact role-partitioned entropy identity

Because M defines a deterministic, disjoint mapping from every species to exactly one role, the implementation satisfies H(S)=U_between+U_within for every valid species posterior. This is an equality, not an empirical approximation.

### Proposition R2: Role sufficiency under within-role ambiguity

If two species posteriors have the same mapped role posterior p_m, they provide the same species-mapped role evidence to the four-role decision. Their U_within values may nevertheless differ. Thus a role decision can remain stable while fine-grained species uncertainty changes; the claimed result is about the fixed label mapping only, not mineral chemistry or visual identifiability in the field.

### Proposition R3: Convex evidence envelope

Since g is constrained to [0,1], every fused role probability is bounded coordinate-wise by the corresponding direct and mapped expert probabilities. The RPG gate changes evidence weighting but cannot create a posterior outside their convex envelope.

### Registered RPG ablations

The formal protocol compares RSG complete against: (i) RPG complete, (ii) RPG without U_within, (iii) RPG without U_between, and (iv) RPG with total species entropy only. All configurations use the same three frozen seeds, residual verifier policy, detached regret-gate branch, data split, and train/test procedures. The ablations test whether the partitioned terms contain evidence not captured by total species entropy alone. Performance conclusions are deferred until the registered formal matrix and paired cluster Bootstrap analysis are complete.

## 5. Capacity-Normalized Monotone Role-Partitioned Gate (Primary Method)

The original RPG gate uses raw natural-log entropy components. However, the four role groups contain unequal numbers of mineral species. A role with more candidate species can produce a larger conditional entropy merely because its support is larger, so raw `U_within` cannot be compared directly with a normalized role entropy or used as a scale-stable gate variable.

Let `n_r` be the number of species mapped to role `r` and let `p_m(r|x)` be the mapped species posterior. We retain the raw exact identity

`H(S|x) = U_between + U_within`.

For gate inputs only, M-RPG defines the available within-role entropy capacity

`C_within(x) = sum_r p_m(r|x) log n_r`,

and the normalized variables

`u_between = U_between / log |R|`,

`u_within = U_within / C_within(x)` when `C_within(x)>0`, and `u_within=0` otherwise.

The direct-role allocation uses a learned base logit that deliberately excludes between-role ambiguity,

`b(x) = Q([h, H(p_d), u_within])`,

and the M-RPG gate is

`g_M(x) = sigmoid(b(x) + softplus(beta) u_between)`.

The final pre-verifier posterior remains

`p_f = g_M p_d + (1-g_M) p_m`.

The use of `softplus(beta)` is not a generic claim that the direct expert is always more reliable. It only enforces a directional allocation rule: holding the shared feature, direct entropy, and within-role uncertainty fixed, rising uncertainty that changes the role decision cannot lower the direct role expert weight.

### Proposition M-R1: Raw entropy identity is preserved

M-RPG computes `U_between` and `U_within` from the same frozen deterministic partition as RPG. Therefore `H(S|x)=U_between+U_within` remains an exact raw-entropy identity. Normalization is applied only after this diagnostic decomposition and does not alter the identity.

### Proposition M-R2: Candidate-capacity normalized range

For valid finite-support posteriors, `0<=u_between<=1` because `H(R|x)<=log|R|`. Further, `0<=U_within<=C_within(x)` by the conditional entropy upper bound inside every role group, so `0<=u_within<=1` whenever the capacity is positive. The explicit zero branch covers images whose positive-mass roles are all singleton groups.

### Proposition M-R3: Monotone direct-expert allocation

For fixed `b(x)` and `u_within`, differentiation gives

`partial g_M / partial u_between = g_M(1-g_M) softplus(beta) >= 0`.

Thus M-RPG cannot reduce direct-role allocation as between-role uncertainty rises. The unconstrained-coefficient ablation removes this property, while the no-between ablation removes this input path. This is a formula-level property of the parameterization, not an empirical performance claim.

### Proposition M-R4: Convex evidence envelope

Because `g_M` is in `[0,1]`, each coordinate of `p_f` lies between the corresponding coordinates of `p_d` and `p_m`; the fused posterior stays on the four-role simplex. The M-RPG gate changes only relative evidence allocation and cannot synthesize a posterior outside both experts' coordinatewise envelope.

### Formal M-RPG evidence and interpretation

The M-RPG protocol uses the same frozen split, augmentations, backbone, optimizer, residual verifiers, RSG regret supervision, and seeds as the RPG protocol. It compares (i) `mrpg_complete`, (ii) `mrpg_unconstrained_between`, which replaces `softplus(beta)` with a signed scalar, and (iii) `mrpg_without_between`, which removes only the between-role term.

Across three seeds, M-RPG obtains 74.34% Macro F1 and 75.83% target-class recall, versus 74.76% and 72.91% for RSG. The paired cluster Bootstrap difference in target recall is +2.92 percentage points with 95% CI [0.13, 5.84], while Macro F1 is -0.41 points with CI [-2.16, 1.30] and Brier difference is inconclusive. Thus the current evidence supports a target-recall-oriented observation, not a general classification or calibration superiority claim.

The direct ablations are also inconclusive: compared with M-RPG, removing monotonicity changes target recall by -0.27 points [ -2.92, 2.52 ], and removing the between-role input changes it by +0.40 points [ -3.45, 5.18 ]. Therefore M-R3 remains a formal guarantee of the parameterization, but the present data do not isolate a stable empirical benefit of its monotone term. This limitation is carried into the paper rather than hidden.

### Registered five-seed extension

Two additional pre-registered seeds (20260730 and 20260731) were run without changing the frozen split, backbone, augmentation, optimizer, RSG residual verifier, or M-RPG formulas. Combining them with the original three seeds, M-RPG obtains 74.03% Macro F1 and 73.47% target-class recall, versus 74.01% and 72.35% for RSG. The paired split-group clustered Bootstrap target-recall difference is +1.12 percentage points with 95% CI [-1.83, 3.90]; Macro F1 is +0.02 points with CI [-1.44, 1.49]. The Ti-bearing-to-target intrusion difference is +0.41 points with CI [-1.79, 2.81], the metallic-to-target intrusion difference is -1.48 points with CI [-5.34, 2.16], and Brier/ECE intervals also cross zero.

Accordingly, the three-seed target-recall observation does not persist as a stable five-seed empirical advantage. The paper may retain M-RPG as a theory-constrained extension with exact propositions M-R1--M-R4 and fully reported negative evidence, but it must not claim target-recall, overall classification, intrusion-control, or calibration superiority over RSG under the current public-specimen protocol.

## 6. Disagreement-Triggered Posterior Calibration (Controlled Ablation)

The Jensen-Shannon disagreement is

D_JS(p_d || p_m) = 1/2 D_KL(p_d || (p_d+p_m)/2) + 1/2 D_KL(p_m || (p_d+p_m)/2).

It determines a bounded correction strength:

rho = 1 - exp[-D_JS(p_d || p_m)].

The calibration representation z_c includes u_d, u_s, u_d elementwise-times u_s, absolute(u_d-u_s), log(p_d)-log(p_m), H(p_d), H(p_m), and D_JS. With residual s=tanh(MLP(z_c)), the calibrated posterior is

p_c = softmax(log p_f + rho s).

The total objective is

L_CGDC = L_HRGV + lambda_dec L_dec + lambda_cal L_cal,

where L_cal = -mean log p_c(y).

## 7. CGDC Formula-Level Propositions

### Proposition P1: Agreement identity

If p_d=p_m, then D_JS(p_d || p_m)=0 and rho=0. Therefore p_c=softmax(log p_f)=p_f=p_d=p_m. Calibration cannot alter an already agreed posterior.

### Proposition P2: Posterior validity

For every input, p_c is the output of softmax. Hence p_c,j>0 and sum_j p_c,j=1.

### Proposition P3: Disagreement-dependent bounded log-odds adjustment

For any roles j and k, because |s_j|<1 and |s_k|<1,

| log[p_c,j/p_c,k] - log[p_f,j/p_f,k] | = rho |s_j-s_k| < 2 rho.

The correction is therefore bounded by the expert disagreement. The bound does not state that the correction improves every prediction; that is an empirical question.

### Proposition P4: Global correction budget

For any two categorical posteriors, the Jensen-Shannon divergence with natural logarithms satisfies 0 <= D_JS <= ln 2. Therefore 0 <= rho = 1-exp[-D_JS] <= 1/2. Combining this with P3 yields a global log-odds adjustment strictly below one for every pair of roles. This prevents the calibration stage from overturning the RSG posterior through an unbounded correction, even under maximal expert disagreement.

## 8. Registered CGDC Empirical Tests

The following five configurations are compared with the same three seeds: RSG complete, CGDC complete, shared-feature CGDC, unconditional CGDC calibration, and CGDC without the decomposition loss.

The registered report includes Accuracy, Macro F1, target recall, Ti-bearing-negative to target intrusion, metallic-hard-negative to target intrusion, Brier score, and expected calibration error. Each difference relative to RSG complete is assessed with paired two-stage Bootstrap: resample random seeds and then complete split_group_id clusters within seed.

## 9. Claim Boundary

Formula-level propositions R1-R3, M-R1-M-R4, and P1-P4 are exact properties of the implemented network. The registered five-seed extension does not establish stable M-RPG gains in target recall, Macro F1, target intrusion, or calibration, and the direct ablations do not establish isolated component causality. Any empirical advantage of RPG, M-RPG, decomposition, or calibration must be stated only when the corresponding paired confidence interval supports it. The stage-conditioned beneficiation decision graph and reject-to-test decision theory remain future work because the current project contains no operating-stage, grade, recovery, material-flow, or assay-cost data.
