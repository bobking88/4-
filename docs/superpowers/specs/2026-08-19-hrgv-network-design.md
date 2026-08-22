# HRGV-Net Network Design Specification

## 1. Purpose and Scope

This specification defines the first-paper network innovation for the vanadium-titanium-magnetite-related mineral image project. The method is named **Hierarchical Risk-Gated Verification Network (HRGV-Net)**. It upgrades the existing hierarchical EfficientNet-B0 classifier without replacing the fixed dataset split, the four-role task, or the existing baselines.

The method addresses an observed failure mode rather than adding a generic attention block: the existing hierarchical model raises target-mineral recall but also raises the rates at which titanium-bearing and metallic hard negatives enter the target stream. HRGV-Net therefore combines two hierarchical role experts and requires a target prediction to pass two asymmetric hard-negative verification heads.

This is a closed-set visual proxy study. It does not estimate mineral grade, vanadium content, recovery, or industrial sorting performance.

## 2. Research Claims

The paper and technical report may claim the following only if supported by the experiments in Section 9:

1. A mineral-species expert and a direct beneficiation-role expert are fused by an input-dependent reliability gate, so the species branch participates in inference instead of acting only as an auxiliary loss.
2. Two asymmetric verification heads are derived from the two measured target-intrusion risks: target versus titanium-bearing negatives and target versus metallic hard negatives.
3. The method changes the target-recall/intrusion trade-off in a measurable way under the fixed test protocol.

The work must not claim that mixture-of-experts, hierarchy learning, hard-negative learning, or selective prediction is generally novel. The defensible contribution is their problem-derived integration for beneficiation-role recognition and the accompanying risk analysis.

## 3. Inputs and Frozen Label Semantics

Input image:

\[
x\in\mathbb{R}^{3\times H\times W}.
\]

Role labels, in the frozen order used by the repository:

1. `target_mineral`;
2. `ti_bearing_negative`;
3. `gangue_negative`;
4. `metallic_hard_negative`.

Species labels and the species-to-role mapping are derived only from the training manifest through `SpeciesRoleMapping`. The mapping matrix is

\[
A\in\{0,1\}^{R\times K},
\]

where each species column maps to exactly one role. The mapping is registered as an immutable model buffer and is saved in every checkpoint.

## 4. Architecture

### 4.1 Shared visual backbone

An ImageNet-pretrained EfficientNet-B0 extracts the pooled representation

\[
h=f_\theta(x)\in\mathbb{R}^{D}.
\]

The same transforms, image size, optimizer family, early-stopping rule, and fixed split used by the existing hierarchical baseline are retained unless an ablation explicitly changes them.

### 4.2 Direct role expert

The direct expert predicts four beneficiation roles:

\[
p_d=\operatorname{softmax}(W_rh+b_r).
\]

### 4.3 Species-mapped role expert

The species head predicts the fine-grained mineral label:

\[
p_s=\operatorname{softmax}(W_sh+b_s).
\]

Its role posterior is obtained with the frozen mapping:

\[
p_m=Ap_s.
\]

The result is normalized after numerical clamping.

### 4.4 Reliability gate

Let \(H(p)\) denote normalized entropy and \(JS(p_d,p_m)\) Jensen-Shannon divergence. The gate is

\[
\alpha(x)=\sigma\left(g_\phi\left([h,H(p_d),H(p_m),JS(p_d,p_m)]\right)\right).
\]

The fused role posterior is

\[
p_f=\alpha(x)p_d+[1-\alpha(x)]p_m.
\]

The gate is a two-layer MLP. It outputs one scalar per image and is trained jointly through the final role loss. No hand-set gate target is used.

### 4.5 Asymmetric hard-negative verifiers

Two binary heads use the shared feature \(h\):

\[
v_{Ti}=P(T\mid T\ \text{or}\ Ti,x),
\qquad
v_{Met}=P(T\mid T\ \text{or}\ Met,x).
\]

The titanium verifier is trained only on target and titanium-bearing samples. The metallic verifier is trained only on target and metallic-hard-negative samples. In each verifier, binary target `1` means target mineral and `0` means its corresponding hard negative. If a mini-batch contains no eligible samples, its verifier loss is a differentiable zero.

The first residual experiment allowed both verifier losses to update the shared backbone. Its three-seed mean target recall remained 2.0 percentage points below the gate-only ablation, which motivates a decoupled verifier path. The revised default supplies stop-gradient features to both verifier heads:

\[
\bar h=\operatorname{stopgrad}(h),
\qquad
z_{Ti}=g_{Ti}(\bar h),
\qquad
z_{Met}=g_{Met}(\bar h).
\]

The verifier heads still learn from their masked binary losses and the final role loss, but these paths cannot directly perturb the visual backbone. A non-decoupled flag is retained as an ablation.

### 4.6 Neutral-zone residual risk correction

The first registered pilot multiplied the fused target evidence by both target-verifier probabilities,

\[
e_T^{mult}=p_f(T)v_{Ti}v_{Met}.
\]

Three-seed evidence showed that this rule systematically attenuated correctly recognized target samples even when both verifiers favored the target. It reduced mean target recall from 74.37% in the gate-only ablation to 71.05%, without improving either hard-negative intrusion mean. This multiplicative rule is therefore retained only as a documented negative pilot.

The revised network uses a neutral-zone contradiction score. For verifier \(h\in\{Ti,Met\}\), threshold \(\tau_h\in(0,1)\), and target probability \(v_h\), define

\[
c_h(v_h)=\frac{[\tau_h-v_h]_+}{\tau_h}.
\]

Only explicit contradiction below the neutral threshold contributes a penalty. The bounded residual scale is

\[
s(x)=\exp[-\beta_{Ti}c_{Ti}(v_{Ti})-\beta_{Met}c_{Met}(v_{Met})],
\]

where \(\beta_{Ti},\beta_{Met}\geq0\). The revised target evidence is

\[
e_T=p_f(T)s(x).
\]

The other role evidence remains unchanged:

\[
e_r=p_f(r),\qquad r\neq T.
\]

The final posterior is

\[
q_r=\frac{e_r}{\sum_j e_j}.
\]

Default values are \(\tau_{Ti}=\tau_{Met}=0.5\) and \(\beta_{Ti}=\beta_{Met}=1\). All training, validation selection, test predictions, confidence values, and downstream selective-recognition analysis use \(q\), not the direct role head alone. The multiplicative pilot remains reproducible through an explicit ablation flag.

## 5. Objective Function

The primary role loss is weighted negative log likelihood on the final posterior:

\[
L_{final}=-w_y\log q_y.
\]

Auxiliary losses are:

\[
L_{direct}=CE(z_r,y_r),
\]

\[
L_{species}=CE(z_s,y_s),
\]

\[
L_{cons}=D_{KL}(p_m\Vert p_d),
\]

\[
L_{verify}=CE(z_{Ti},y_{Ti})+CE(z_{Met},y_{Met}),
\]

and the existing role-aware supervised contrastive term \(L_{contrast}\).

The complete objective is

\[
L=L_{final}
+\lambda_dL_{direct}
+\lambda_sL_{species}
+\lambda_cL_{cons}
+\lambda_vL_{verify}
+\lambda_hL_{contrast}.
\]

Default weights for the first controlled experiment are:

- \(\lambda_d=0.25\);
- \(\lambda_s=0.50\);
- \(\lambda_c=0.10\);
- \(\lambda_v=0.50\);
- \(\lambda_h=0.10\).

Weights may be changed only from the validation split and every change must be recorded in `environment.json`.

## 6. Theoretical Results Used in the Report

### Proposition 1: convex fusion log-loss bound

For the true role \(y\), \(0\leq\alpha\leq1\), and positive expert probabilities,

\[
-\log[\alpha p_d(y)+(1-\alpha)p_m(y)]
\leq
\alpha[-\log p_d(y)]+(1-\alpha)[-\log p_m(y)].
\]

This follows from convexity of \(-\log\). It bounds the mixture loss by the weighted expert losses; it does not imply that the mixture always beats each individual expert.

### Proposition 2: neutral-zone identity and bounded monotonic correction

If both verifiers accept the target at their neutral thresholds,

\[
v_{Ti}\geq\tau_{Ti},\qquad v_{Met}\geq\tau_{Met},
\]

then \(c_{Ti}=c_{Met}=0\), \(s(x)=1\), and

\[
q=p_f.
\]

Thus, unlike direct probability multiplication, supportive verifier outputs cannot suppress target evidence. For fixed \(p_f\), lowering either verifier probability below its threshold cannot increase \(q_T\). Because \(0\leq c_h\leq1\), the target evidence scale is bounded by

\[
\exp[-(\beta_{Ti}+\beta_{Met})]\leq s(x)\leq1.
\]

### Proposition 3: verifier false-acceptance upper bound

Let target acceptance for a hard-negative type \(h\in\{Ti,Met\}\) require both base target evidence and the corresponding verifier:

\[
\mathcal A_T=\{p_f(T)\geq\tau\}\cap\{v_h\geq\eta_h\}.
\]

Then

\[
P(\mathcal A_T\mid Y=h)
\leq
\min\left\{
P[p_f(T)\geq\tau\mid Y=h],
P[v_h\geq\eta_h\mid Y=h]
\right\}.
\]

This is the probability-of-an-intersection bound and needs no independence assumption. The empirical counterpart is the hard-negative-to-target intrusion rate.

### Proposition 4: selective risk compatibility

The final posterior \(q\) remains a normalized categorical posterior and can be calibrated on the validation split. Existing risk-coverage and defer rules therefore apply without changing the fixed test labels. This is a compatibility property, not a new rejection theorem.

### Proposition 5: verifier-gradient isolation

Let \(\theta\) denote backbone parameters, \(h=f_\theta(x)\), and \(L_{verify}\) the sum of the two masked verifier losses. Under \(\bar h=\operatorname{stopgrad}(h)\),

\[
\frac{\partial L_{verify}}{\partial \theta}
=
\frac{\partial L_{verify}}{\partial \bar h}
\frac{\partial\operatorname{stopgrad}(h)}{\partial h}
\frac{\partial h}{\partial\theta}
=0.
\]

This does not assert that all task interference disappears: the final role loss still updates the shared role and species paths. It proves only that the auxiliary verifier objectives cannot directly rotate the backbone representation.

## 7. Numerical and Behavioral Invariants

1. `direct_role_probabilities`, `mapped_role_probabilities`, `fused_role_probabilities`, and `final_role_probabilities` have shape `[batch, 4]`, are finite, non-negative, and sum to one within tolerance `1e-6`.
2. Gate values have shape `[batch, 1]` and lie in `[0, 1]`.
3. With gate equal to one, fused probabilities equal direct probabilities; with gate equal to zero, they equal mapped probabilities.
4. With both verifier target probabilities at or above their neutral thresholds, final probabilities equal fused probabilities.
5. Lowering either verifier target probability below its threshold cannot increase the final target probability while fused probabilities are fixed.
6. The residual target-evidence scale remains within its analytical bounds.
7. The two verifier losses backpropagate only through eligible samples.
8. With verifier decoupling enabled, a verifier-only loss produces gradients in both verifier heads and zero gradient in the backbone.
9. The complete model supports a backward pass through the backbone, gate, role head, species head, both verifiers, and projection head.

## 8. Output Contract

Each run writes:

- `environment.json` with architecture, mapping, verifier mode, decoupling state, neutral thresholds, strengths, all loss weights, seed, and dataset paths;
- `best_model.pt` with state dict, role labels, species labels, mapping, and best validation Macro F1;
- `metrics_history.csv` with every loss term, Macro F1, accuracy, mean gate value, and mean expert divergence;
- `test_metrics.json` with four-class metrics, target proxy metrics, species accuracy, verifier accuracies on their eligible subsets, mean gate value, and mean expert divergence;
- `test_predictions.csv` with final prediction/confidence plus direct prediction, mapped prediction, gate, both verifier target probabilities, and expert divergence;
- `confusion_matrix.csv`.

## 9. Experimental Matrix and Acceptance Criteria

Every formal configuration uses seeds `20260727`, `20260728`, and `20260729`.

Required comparisons:

1. existing weighted-CE EfficientNet-B0 baseline;
2. existing hierarchical role-aware model;
3. HRGV without verifiers (gate only);
4. HRGV without learned gate (equal expert fusion);
5. HRGV without contrastive loss;
6. complete decoupled residual HRGV-Net;
7. residual HRGV-Net without verifier-gradient decoupling;
8. the registered multiplicative-verifier pilot.

Primary outcomes:

- target recall and target miss rate;
- titanium-bearing-negative to target intrusion rate;
- metallic-hard-negative to target intrusion rate;
- Macro F1 and accuracy.

Secondary outcomes:

- species accuracy;
- gate distribution by actual role;
- direct-versus-mapped disagreement;
- verifier ROC-AUC on eligible subsets;
- risk-coverage curve after validation-only calibration.

The method is considered to improve the current risk trade-off only if the three-seed mean reduces both hard-negative intrusion rates relative to the existing hierarchical model while target recall decreases by no more than 1.0 percentage point. Macro F1 and accuracy are reported regardless of outcome. Paired cluster bootstrap confidence intervals and paired exact tests are required; a favorable point estimate alone is insufficient.

## 10. Related-Work Boundaries

Recent work already covers bidirectional hierarchical logits, structural hierarchies for partial labels, graph-based hierarchical material recognition, and post-hoc metric-specific hierarchical decoding. HRGV-Net must therefore be presented as a beneficiation-risk-derived architecture, not as the first hierarchical or gated classifier.

- BiLT: <https://arxiv.org/abs/2412.12782>
- SHIP: <https://openaccess.thecvf.com/content/WACV2025/html/Kadam_SHIP_Structural_Hierarchies_for_Instance-Dependent_Partial_Labels_WACV_2025_paper.html>
- Hierarchical Material Recognition from Local Appearance: <https://openaccess.thecvf.com/content/ICCV2025/html/Beveridge_Hierarchical_Material_Recognition_from_Local_Appearance_ICCV_2025_paper.html>
- Metric-specific hierarchical decoding: <https://openreview.net/forum?id=5zsBvPOIUQ&noteId=28HBqiEt3W>

## 11. Reporting Language

If the acceptance criterion is met, the report may state that HRGV-Net improved the measured recall-intrusion trade-off on the fixed public-specimen test set. If it is not met, the report must state which component changed which risk and treat the network as a controlled negative or mixed result. In neither case may the result be interpreted as industrial recovery, grade, or production-line performance.
