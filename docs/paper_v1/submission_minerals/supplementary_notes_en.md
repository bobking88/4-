# Supplementary Notes: Regret-Supervised Cross-Granularity Routing

> Working supplement, 15 November 2026; baseline and provenance update, 17 November 2026. This document translates and reorganizes previously archived definitions, proofs and diagnostics. It introduces no experiment, model, numerical recomputation or claim of priority. Equation labels R9a-R9e retain their Chinese-source identifiers; S-labels below are organizational labels, not additional original theorems. See the [manuscript](manuscript_en.md) and [source map](README.md).

## S1. Exact Batch Loss Definitions

### S1.1. Classification and Consistency

Let $\mathcal B$ denote a batch, with role label $y_i$ and species label $s_i$. For positive training-class counts $n_c$, the mean-normalized inverse-frequency weight is $\omega_c=(1/n_c)/(K^{-1}\sum_{k=1}^{K}1/n_k)$. Role and species weights use their respective training counts. In real-probability notation,

$$
\begin{aligned}
\mathcal C_\omega(p,y;\mathcal B)
&=-\frac{\sum_{i\in\mathcal B}\omega_{y_i}\log p_{i,y_i}}
{\sum_{i\in\mathcal B}\omega_{y_i}},\\
\mathcal L_{role}&=\mathcal C_{\omega^r}(q,y;\mathcal B),\qquad
\mathcal L_{direct}=\mathcal C_{\omega^r}(p_d,y;\mathcal B),\\
\mathcal L_{species}&=\mathcal C_{\omega^s}(p_s,s;\mathcal B).
\end{aligned}
\tag{R9a}
$$

The denominator is the sum of the observed label weights, not the number of images. The consistency term uses a batch mean and the mapped-to-direct KL direction:

$$
\mathcal L_{KL}=\frac1{|\mathcal B|}\sum_{i\in\mathcal B}
\sum_{c=1}^{4}p_{m,ic}\log\frac{p_{m,ic}}{p_{d,ic}}.
\tag{R9b}
$$

### S1.2. Masked Verification

For $h\in\{Ti,Met\}$, define $\mathcal B_h=\{i\in\mathcal B:y_i\in\{T,h\}\}$ and $b_i=\mathbf1[y_i=T]$. Given verifier target probability $v_{h,i}$,

$$
\mathcal L_h=-\frac1{|\mathcal B_h|}\sum_{i\in\mathcal B_h}
\left[b_i\log v_{h,i}+(1-b_i)\log(1-v_{h,i})\right].
\tag{R9c}
$$

Each verifier uses unweighted binary cross-entropy. An empty subset returns a zero connected to the computation graph. The two subset means are added; the subsets are not pooled into one mean. The index $h$ here identifies a verifier, not the feature vector in R1.

### S1.3. Role-Aware Contrast

Let $e_i$ be a normalized projected feature and $\kappa_{ij}=e_i^\top e_j/T_{con}$. All indices below refer to the current batch:

$$
\begin{gathered}
P_i=\{j\in\mathcal B\setminus\{i\}:y_j=y_i\},\qquad
I=\{i\in\mathcal B:P_i\ne\varnothing,\ N_i\ne\varnothing\},\\
N_i=\begin{cases}
\{j:y_j\in\{Ti,Met\}\},&y_i=T,\\
\{j:y_j=T\},&y_i\in\{Ti,Met\},\\
\varnothing,&\text{other roles},
\end{cases}
\end{gathered}
\tag{R9d}
$$

$$
\mathcal L_{contrast}=-\frac1{|I|}\sum_{i\in I}
\log\frac{\sum_{j\in P_i}\exp(\kappa_{ij})}
{\sum_{j\in P_i\cup N_i}\exp(\kappa_{ij})}.
\tag{R9e}
$$

The loss is zero when $I$ is empty or the batch has fewer than two images. Positive exponential scores are summed before the logarithm; replacing this with an average of individual positive-pair log losses changes the objective. Gangue is neither an anchor nor a negative in this term. Titanium and metallic interference are not direct negatives for one another. Main classification and other losses still supervise these relationships. These definitions describe the existing component and do not introduce a new contrastive objective.

Numerical clipping and graph-zero conventions remain implementation details of the archived source. Historical per-run source identities are incomplete (S5); the equations do not retrospectively certify every training binary.

## S2. Supporting Conditional Properties

### S2.1. Convex Fusion and Target Odds

For positive true-class expert probabilities $a,b$ and $g\in[0,1]$, convexity gives

$$
-\log[ga+(1-g)b]\le g(-\log a)+(1-g)(-\log b).
\tag{S1}
$$

The comparator is a weighted average of expert losses, not necessarily the better expert. This standard Jensen bound does not guarantee superior accuracy.

The posterior correction in R6-R8 satisfies, for any non-target role $j$ with positive probability,

$$
\frac{q_T}{q_j}=\frac{p_{f,T}}{p_{f,j}}e^{-\eta},\qquad
\frac{\partial}{\partial c_h}\log\frac{q_T}{q_j}=-\beta_h\le0.
\tag{S2}
$$

The derivative holds with the fused probabilities and other penalties fixed. If $\eta=0$, the map is the identity. Non-target odds $q_j/q_k=p_{f,j}/p_{f,k}$ are preserved. This exact directional property cannot establish whether suppression is appropriate for a particular mineral image.

For a separate strict-acceptance rule requiring an initial acceptance event $E_0$ and verifier event $E_h$, $P(E_0\cap E_h)\le\min\{P(E_0),P(E_h)\}$. Writing their joint probability as a product requires independence. This event-containment observation is not a theorem that the residual correction R6-R8 implements strict conjunction or improves recall.

### S2.2. Exact Regret, Local Bound and Units

With $g_o=\mathbf1[a\ge b]$, $M=\max(a,b)$, $d=|a-b|$ and $\delta=|g-g_o|$, the case $a\ge b$ gives $ga+(1-g)b=a-(1-g)(a-b)$. The other case gives $b-g(b-a)$. Both yield

$$
p_g=M-\delta d,\qquad r=-\log(1-\delta d/M).
\tag{S3}
$$

If $a,b\ge\varepsilon>0$, the mean value theorem for $-\log t$ gives $0\le r\le\delta d/\varepsilon$. Taking the sample-specific floor $\varepsilon_i=\min(a_i,b_i)$ yields the archived diagnostic bound

$$
r_i\le B_i^{loc}
=\frac{|g_i-g_{o,i}|\,|a_i-b_i|}{\min(a_i,b_i)}.
\tag{S4}
$$

For fixed $M$ and feasible $\delta d<M$, the partial derivatives are $d/(M-\delta d)$ and $\delta/(M-\delta d)$. They are nonnegative, and strictly positive only when the other factor is positive. This is not an ordering across samples with different $M$. The local bound includes quantities used to define regret; stratifying samples by it is descriptive mechanism analysis, not independent prediction of errors.

All logarithms in these loss expressions are natural, so regret and NLL differences are in nat. Multiplying a loss difference by 100 for display does not convert it to a percentage-point accuracy difference. Bounds explain the risk being measured, not the novelty or empirical optimality of the tanh weighting in R11.

### S2.3. Soft Targets and Entropy Regularization

For a fixed labelled sample and $T>0$, consider

$$
J_T(t)=t\ell_d+(1-t)\ell_m
+T[t\log t+(1-t)\log(1-t)],\quad t\in[0,1].
\tag{S5}
$$

Using $0\log0=0$, its interior derivatives are

$$
J_T'(t)=\ell_d-\ell_m+T\log\frac{t}{1-t},\qquad
J_T''(t)=\frac{T}{t(1-t)}>0.
\tag{S6}
$$

Strict convexity and the endpoint derivative limits give the unique solution $t^*=\sigma((\ell_m-\ell_d)/T)$. This is a two-expert instance of standard entropy-regularized selection, not an independent first-principles invention. Regularized probability mappings are discussed, for example, by Niculae and Blondel [SRef1]. The algebra here is supplied to explain the implemented target, not to attribute this particular mineral construction to that paper.

The two relevant KL identities must not be conflated:

$$
J_T(t)-J_T(t^*)=T D_{KL}(\operatorname{Ber}(t)\Vert\operatorname{Ber}(t^*)),
\tag{S7}
$$

$$
\operatorname{BCE}(g,t^*)-H(t^*)
=D_{KL}(\operatorname{Ber}(t^*)\Vert\operatorname{Ber}(g)).
\tag{S8}
$$

They have the same pointwise minimizer but different KL directions, objectives and gradients. BCE imitates the oracle target; it is not an equivalent implementation of directly minimizing $J_T$. Likewise, the linear expert loss in S5 differs from the log loss of fused probabilities. A finite-capacity gate shared across samples need not attain every samplewise optimum. At an expert tie, $w=0$, so the regret term does not uniquely force $g=1/2$.

### S2.4. Gradient Isolation and Selective Decisions

With nonshared gate parameters, stopping gradients through the full gate input, soft target and weight removes all paths from $\mathcal L_{reg}$ to expert and backbone parameters. The R12 denominator contains only detached weights, and neither clipping nor a zero-weight branch restores a path. This proves T3, including its batch implementation. A nonzero derivative with respect to the gate output is insufficient to guarantee a parameter update: gate Jacobians, batch cancellation or clipping can yield a zero parameter gradient. Other task losses still train the shared features.

A separate decision-theoretic interpretation is available only under explicit assumptions. For true conditional posterior $\pi(y\mid x)$, known action costs $C(y,a)$ and constant total rejection loss $c_{rej}$, define

$$
R_{act}(x)=\min_a\sum_y C(y,a)\pi(y\mid x).
\tag{S9}
$$

Pointwise comparison with $c_{rej}$ gives the Bayes-optimal choice between an ordinary action and rejection. Substituting estimated network probabilities minimizes estimated risk, not automatically true risk. If rejection means laboratory testing, its loss includes residual post-test decision loss as well as test cost. No measured assay costs or process-stage decision labels support an industrial optimum here. This standard conditional rule is background, not a completed method contribution or a new stage-conditioned experiment.

## S3. Limits of Soft Supervision and Scalar Fusion

### S3.1. Population BCE Is Not Fused NLL

Fix $x$ and the expert vectors, let $\pi_y$ be the true conditional label distribution, and let $t_y,w_y$ be their label-dependent soft target and weight. For positive total weight, the population-ratio objective is

$$
Q_x(g)=\frac{\sum_y\pi_yw_y[-t_y\log g-(1-t_y)\log(1-g)]}
{\sum_y\pi_yw_y}.
\tag{S10}
$$

Writing $\mu_x=\sum_y\pi_yw_yt_y/\sum_y\pi_yw_y$ gives

$$
Q_x'(g)=\frac{g-\mu_x}{g(1-g)},\qquad g^*_{BCE}=\mu_x.
\tag{S11}
$$

The minimizer is unique for $0<\mu_x<1$; all-zero weights impose no gate constraint. This population ratio is not, without further justification, the expectation of the random batch-normalized R12 loss. Meanwhile,

$$
R_x(g)=-\sum_y\pi_y\log[b_y+g(a_y-b_y)],\qquad
R_x'(g)=-\sum_y\frac{\pi_y(a_y-b_y)}{b_y+g(a_y-b_y)}.
\tag{S12}
$$

The two optimality conditions generally differ. The existing four-role counterexample uses

$$
\pi=(.4,.4,.1,.1),\quad a=(.72,.08,.1,.1),\quad b=(.48,.32,.1,.1),
\quad T_r=.2,\quad T_w=.5.
\tag{S13}
$$

Its archived $g^*_{BCE}$ is 0.3567875763. For $g\in[0,1]$, the first two fused probabilities are $p_1=.48+.24g$ and $p_2=.32-.24g$, with $p_1>p_2$. Therefore

$$
R_x'(g)=.096(1/p_2-1/p_1)>0,
\tag{S14}
$$

and the fused-NLL minimum is at $g^*_{NLL}=0$, not $g^*_{BCE}$. Archived NLL values are 1.2098784019 and 1.2687837736 nat, respectively, a difference of 0.0589053717 nat. These numbers are transcribed from the existing [theory appendix](../theoretical_appendix.md), not recomputed for this translation. The derivative proves the optimum; the previously archived grid check is only a numerical cross-check.

This counterexample excludes an unconditional NLL-consistency claim for the isolated soft-target objective. It does not identify the true mineral-image conditional distribution or show that the full multi-loss network must degrade. Changing the supervision split alone cannot remove the algebraic non-equivalence.

### S3.2. Fixed-Endpoint Classification Capacity

Consider $q_\rho=(1-\rho)q_0+\rho q_1$, $\rho\in[0,1]$, with both endpoints fixed for each image. If both endpoints have the same unique winning class $k$, then $q_{0,k}-q_{0,j}>0$ and $q_{1,k}-q_{1,j}>0$ for every $j\ne k$. Their convex combination remains positive, so scalar interpolation cannot change that winning class. Under unique endpoint winners, the finite-sample bound is

$$
|\operatorname{Acc}(q_\rho)-\operatorname{Acc}(q_0)|
\le \frac1n\sum_{i=1}^n
\mathbf1[\arg\max q_{0,i}\ne\arg\max q_{1,i}].
\tag{S15}
$$

The same argument applies if the scalar weight varies by image. Tie handling must be specified if uniqueness is dropped. Target-recall improvement can occur only among fallback-missed targets whose endpoint decisions disagree. This is a capacity limit for fixed endpoints, not a generalization theorem or a limit for retraining the endpoints. It also does not say that continuous fusion can never be correct when both individual experts are wrong but disagree with each other.

## S4. Negative and Development Evidence

The main paper does not select a successful story by omitting later failures. The following results have different evidential status and are not pooled. NLL entries use nat; classification differences use percentage points.

| Archived diagnostic | Existing observation | Interpretation and limit |
|---|---|---|
| Fixed-expert substitution | Learned-minus-equal-weight NLL differences: fixed split -0.002537, +0.000371, -0.000394; photographer holdout -0.001756, +0.001200, -0.000452; ResNet50 +0.000126, +0.000024, -0.000293 | Nine post-hoc comparisons have inconsistent signs; no stable fixed-expert advantage is established. |
| Gate-only retraining | On one frozen expert checkpoint, regret BCE versus equal weighting reduced validation NLL by 0.001381 and reduced accuracy by 0.1817 points | Three gate initializations share one expert; validation was reused for selection, so they are not three independent expert confirmations. |
| Verifier-aware objective | Final NLL plus regret BCE versus equal weighting reduced validation NLL by 0.000713, but macro F1 was lower by 0.1262 points | Exploratory shared-checkpoint comparison; selecting a theoretically aligned objective did not establish overall generalization benefit. |
| Out-of-sample routing confirmation | Relative to equal weighting: NLL -0.014040 [-0.019431, -0.008724]; target recall -2.476 [-4.381, -0.571] points | Independent expert-seed confirmation improved probability risk but breached the preset one-point recall-harm limit; the candidate was not promoted. |
| TC and ABMP development | TC did not establish a beneficial projection effect; all 27 ABMP development candidates failed their effectiveness gate, with zero target-recall increase capacity for the tested endpoints | Further scalar routing over the same endpoints is not evidence of new classification capacity. |
| LC-RFA and IC-RFA development | No net role-branch classification correction was established; 92/108 existing IC anchor errors were excluded by the old output budget | Branch activity or a small NLL movement does not establish a useful new visual mechanism. |
| MSB-RRE candidate | Implementation and synthetic checks exist, but real mechanism effectiveness remains unvalidated | It is not part of the completed results or a stronger-network claim. No additional candidate experiment is scheduled for the first paper. |

For a fixed checkpoint, $L_{fused}=L_{oracle}+r$. Across separately trained models, $\Delta L_{fused}=\Delta L_{oracle}+\Delta r$. Consequently, cross-model regret improvement does not isolate a gate-only effect; the expert reference can also change.

The out-of-sample confirmation used three independent expert seeds and 10,000 two-stage paired bootstrap resamples, first by expert seed and then by duplicate-image group within species-role strata. It is distinct from the three main routing protocols and their resampling counts. The relevant final evaluation set has already been exposed and is not available as a new untouched confirmation set. Earlier gate-only explorations reused expert-stop or validation data and have weaker status. See the [development archive](../development_diagnostics_appendix.md) for the original comparison directions, splits and nonpromotion decisions.

## S5. Reproducibility Record and Remaining Inputs

### S5.1. Archived Settings

The following settings are transcribed from the existing [configuration review](../reproducibility_materials.md) of 18 formal environment records; no new model or data inspection is performed here.

| Setting | HRGV reference | RSG-HRGV |
|---|---|---|
| Gate hidden / projection output dimensions | 128 / 128 | 128 / 128 |
| Direct / species / KL coefficients | 0.25 / 0.50 / 0.10 | Same |
| Verifier / contrastive coefficients | 0.50 / 0.10 | Same |
| Regret coefficient | 0 | 0.10 |
| Regret target / gap temperatures | 0.20 / 0.50, branch disabled | 0.20 / 0.50 |
| Contrastive temperature | 0.10 | 0.10 |
| Gate input detached | No | Yes |
| Soft target / gap weight | Branch disabled | Both enabled |
| Verification mode | Residual, enabled | Same |
| Both verifier thresholds / strengths | 0.50 / 1 | Same |
| Verifier feature detached | No | No |
| Device / PyTorch record | CUDA / 2.12.1+cu130 | Same |

The source review includes a [snapshot with original-record hashes](../materials_2026-11-03/run_configuration_snapshot.json). This is a configuration excerpt, not a complete archive of every original environment file or a certificate of run completion. Structural dropout values and layer dimensions come from the archived architecture description, not a claim that all were recorded per run.

### S5.2. Unresolved Historical Details

| Item | What is supported | What must not be inferred |
|---|---|---|
| Initialization | Historical notes describe ImageNet-pretrained EfficientNet-B0 and ResNet50 IMAGENET1K_V2 | Per-run weight hashes or identical preprocessing solely from a model name |
| Selection | Notes identify validation macro F1 early stopping and matched strategies | Exact historical epoch limit, patience or minimum improvement for all runs |
| Optimizer and batch | Not recorded in the 18 selected environment records | Optimizer, learning rate, decay or batch size taken from current defaults or later experiments |
| Input transforms | Notes describe matched augmentation | Exact dimensions, crop, normalization or augmentation parameters without historical evidence |
| Hardware and dependencies | CUDA and the recorded PyTorch string | GPU model, memory, driver, cuDNN, Python or a full lockfile; a current hardware probe cannot repair a historical record |
| Source identity | Current documentation and file hashes can be frozen | A modern documentation commit is not the commit used for every old training run |
| Data and weights | Image manifests, provenance and some result artifacts are archived | All images are redistributable, all weights are supplied, or every original URL remains accessible |

### S5.3. Bounded Historical Trace

A follow-up audit completed on 17 November 2026 examined the three main RSG protocol notes, the earlier HRGV note, existing training instructions, metadata-file identities and the available ResNet50 stdout/stderr logs. All 18 selected environment files matched their snapshot hashes. The fixed-split and ResNet50 documentation gives runner commands, but neither records explicit batch size, learning rate, decay or a resolved full argument set. The photographer-holdout note specifies the paired flags and shared training policy, not a complete per-run command. The available ResNet50 stdout includes skips of already completed runs, epoch progress and early-stopping messages; it does not supply the missing optimizer configuration. The stderr file is empty. A stopping epoch is not a record of the configured epoch limit or patience.

The earlier baseline instructions specify an example epoch limit of 30 and batch size of 16, and the baseline summaries name AdamW. Those are historical instructions or family-level descriptions, not evidence that every subsequent RSG run used those exact settings. No value was promoted from a current default or a later experiment into the historical configuration table. The bounded trace therefore strengthens provenance reporting but does not close the missing fields in S5.2. Its inputs and file hashes are listed in the [audit record](archive_trace_2026-11-17.md).

These are substantive reproducibility limitations, not merely missing typography. The one-pass archive trace is complete within its stated scope; repeating the same search or training again cannot repair an old execution record. Exact historical settings may be added only if an identifiable original record is supplied. A self-contained journal supplement and a frozen public artifact inventory are still required. Author identity, funding, conflicts and approval cannot be inferred from the repository and remain author-supplied information. Disclosure of the gaps is not assurance that an editor or reviewer will regard them as acceptable.

## S6. Historical Baseline Provenance

Table 5 transcribes existing three-seed aggregate results; it does not recompute scores from predictions. The dataset version is `dataset_final_v1`, with the fixed 5,961/1,284/1,284 split and seeds 20260727, 20260728 and 20260729 according to the archived notes. Row-specific provenance is as follows:

| Main Table 5 rows | Archived source | Method distinction |
|---|---|---|
| Plain ResNet50 and EfficientNet-B0 | [Formal baseline summary](../../../outputs/training/formal_experiment_summary_v1.md) | ImageNet initialization, weighted cross-entropy; historical summary names AdamW and early stopping. |
| Weighted Focal Loss | [Three-seed Focal note](../../experiment_records/2026-07-30_focal_loss_three_seed_ablation.md) | The note describes replacing weighted cross-entropy with weighted Focal Loss, gamma 2, under matched within-study settings. |
| Role-aware precursor | [Role-aware experiment note](../../experiment_records/2026-07-30_role_aware_hard_negative_learning.md) | Four-role head, auxiliary binary head and 128-dimensional projection; loss weights 1, 0.25 and 0.10. |
| Hierarchical precursor | [Hierarchical aggregate table](../../../outputs/business_metrics/hierarchical_efficientnet_b0/hierarchical_metrics_three_seed.md) and the preceding role-aware note | Adds the 17-species head and species-to-role aggregation. Its consistency direction is role-to-mapped KL, unlike the mapped-to-direct KL in R9b; it is not merely HRGV without a gate. |
| Coupled residual HRGV and RSG-HRGV | [RSG fixed-split note](../../experiment_records/2026-08-22_rsg_hrgv.md) and main Table 1 | Shared inference topology; training differs in regret supervision and gate-input detachment. |

This table is limited to already completed baselines. It neither reproduces the cited external papers on a common dataset nor constitutes a comparison with current state-of-the-art architectures. The earlier [HRGV record](../../experiment_records/2026-08-22_hrgv_network.md) states that plain-CNN per-image predictions were not saved; only their seed summaries are used here. No paired interval is reconstructed from those summaries.

There is an unresolved reporting discrepancy outside the two selected Table 5 metrics: for the same plain EfficientNet-B0 target-recall mean of 71.98%, the old class-recall table reports a sample standard deviation of 0.83 percentage points, whereas the binary proxy table reports 0.46. If both summarize the same target labels, predictions and seeds, the recall statistic should coincide. The archived tables alone do not establish the cause. Table 5 therefore does not copy target-recall dispersion from either version; this is an explicitly disclosed exclusion, not a correction of the historical record. The discrepancy does not concern the separate HRGV/RSG rows in Table 1. Raw predictions and new calculations were not used to resolve it in this non-experimental consolidation.

## Supplementary Reference

SRef1. Niculae, V.; Blondel, M. A Regularized Framework for Sparse and Structured Neural Attention. *Advances in Neural Information Processing Systems* **2017**, *30*. [Publisher-hosted paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/2d1b2a5ff364606ff041650887723470-Paper.pdf). This existing background citation supports the regularization context; it does not establish priority for the present application.
