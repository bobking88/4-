# CGDC-RSG-HRGV-Net: Methods and Theoretical Statements

## Intended Paper Positioning

This paper studies closed-set visual recognition of vanadium-titanium-magnetite-related minerals from publicly available mineral specimen images. It does not infer V, Ti, or Fe grade, recovery, industrial sorting performance, or mineral chemistry from RGB images.

The central contribution is a cross-granularity evidence model for a fixed four-role task. The model is evaluated as an extension of RSG-HRGV, using the same data split, ImageNet-pretrained EfficientNet-B0 backbone, and three registered random seeds.

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

## 4. Disagreement-Triggered Posterior Calibration

The Jensen-Shannon disagreement is

D_JS(p_d || p_m) = 1/2 D_KL(p_d || (p_d+p_m)/2) + 1/2 D_KL(p_m || (p_d+p_m)/2).

It determines a bounded correction strength:

rho = 1 - exp[-D_JS(p_d || p_m)].

The calibration representation z_c includes u_d, u_s, u_d elementwise-times u_s, absolute(u_d-u_s), log(p_d)-log(p_m), H(p_d), H(p_m), and D_JS. With residual s=tanh(MLP(z_c)), the calibrated posterior is

p_c = softmax(log p_f + rho s).

The total objective is

L_CGDC = L_HRGV + lambda_dec L_dec + lambda_cal L_cal,

where L_cal = -mean log p_c(y).

## 5. Formula-Level Propositions

### Proposition P1: Agreement identity

If p_d=p_m, then D_JS(p_d || p_m)=0 and rho=0. Therefore p_c=softmax(log p_f)=p_f=p_d=p_m. Calibration cannot alter an already agreed posterior.

### Proposition P2: Posterior validity

For every input, p_c is the output of softmax. Hence p_c,j>0 and sum_j p_c,j=1.

### Proposition P3: Disagreement-dependent bounded log-odds adjustment

For any roles j and k, because |s_j|<1 and |s_k|<1,

| log[p_c,j/p_c,k] - log[p_f,j/p_f,k] | = rho |s_j-s_k| < 2 rho.

The correction is therefore bounded by the expert disagreement. The bound does not state that the correction improves every prediction; that is an empirical question.

## 6. Registered Empirical Tests

The following five configurations are compared with the same three seeds: RSG complete, CGDC complete, shared-feature CGDC, unconditional CGDC calibration, and CGDC without the decomposition loss.

The registered report includes Accuracy, Macro F1, target recall, Ti-bearing-negative to target intrusion, metallic-hard-negative to target intrusion, Brier score, and expected calibration error. Each difference relative to RSG complete is assessed with paired two-stage Bootstrap: resample random seeds and then complete split_group_id clusters within seed.

## 7. Claim Boundary

Formula-level propositions P1-P3 are exact properties of the implemented network. Any empirical advantage of decomposition or calibration is limited to the formal public-specimen four-role protocol and must be stated only when the corresponding paired confidence interval supports it. The stage-conditioned beneficiation decision graph and reject-to-test decision theory remain future work because the current project contains no operating-stage, grade, recovery, material-flow, or assay-cost data.
