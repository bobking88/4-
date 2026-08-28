# Monotone Role-Partitioned Granularity Gate (M-RPG-HRGV) Design

## Purpose

M-RPG-HRGV is the final proposed network for the first paper. It upgrades
RPG-HRGV from a free-form entropy-conditioned gate to a role-partition-aware
gate with a verifiable monotonic response to *between-role* uncertainty. The
network is evaluated only for closed-set four-role visual recognition on
public mineral specimen images. It does not estimate mineral chemistry,
grade, recovery, industrial sorting performance, or operating-stage action.

## Problem Structure

The current dataset has a frozen deterministic map from 17 mineral species
`S` to four beneficiation-role labels `R=phi(S)`. Let `M[r,k]=1` iff
`phi(k)=r`. The shared EfficientNet-B0 feature is `h=f_theta(x)`, the direct
role expert outputs `p_d(r|x)`, and the mineral-species expert outputs
`p_s(k|x)`. The mapped role evidence is

`p_m(r|x) = (M p_s)_r`.

The original RPG identity remains exact in natural-log units:

`H(S|x) = H(R|x) + H(S|R,x) = U_between + U_within`.

`U_between=H[p_m]` captures uncertainty that changes the four-role decision;
`U_within=H(S|R,x)` captures uncertainty among species that already belong to
the same role. This partition-specific meaning, rather than generic use of a
single entropy scalar or generic mixture-of-experts routing, is the proposed
task-specific contribution.

## Candidate-Capacity Normalization

The four roles have unequal species cardinalities. Comparing raw conditional
entropy with direct-role entropy therefore mixes different numerical scales.
M-RPG preserves the raw chain-rule identity above for theoretical analysis,
but feeds normalized reliability variables to the gate.

Let `n_r=|phi^{-1}(r)|` be the number of species assigned to role `r` and

`C_within(p_m)=sum_r p_m(r|x) log n_r`.

Define

`u_between = H[p_m]/log |R|`,

`u_within = H(S|R,x)/C_within(p_m)` when `C_within(p_m)>0`, and zero
otherwise. Both values lie in `[0,1]`. The proof follows from the entropy
upper bounds `H(R)<=log |R|` and
`H(S|R,x)<=sum_r p_m(r|x) log n_r`.

This normalization is not presented as a new entropy identity. It is a
partition-aware gate parametrization that prevents a role with more candidate
species from receiving a larger uncertainty value solely because its label set
is larger.

## Monotone Gate

M-RPG retains the direct role expert, species expert, residual Ti-bearing and
metallic hard-negative verifiers, and existing soft gate-regret supervision.
It changes only the gate. Let

`b(x)=Q([h, H(p_d), u_within])`

be a learned scalar that deliberately excludes `u_between`. The direct-role
weight is

`g_M = sigmoid(b(x) + softplus(beta) u_between)`.

The fused four-role posterior is

`p_f = g_M p_d + (1-g_M) p_m`.

The proposed inductive bias is explicit: if mapped species evidence becomes
less certain *between roles*, the direct role expert cannot receive a smaller
weight, holding the other evidence fixed. This does not assume that the direct
expert is always correct; it constrains only the direction by which the gate
responds to role-relevant ambiguity.

## Formula-Level Statements

### M-R1: Partitioned entropy identity

The raw quantities obey `H(S|x)=U_between+U_within` exactly for the frozen
species-to-role mapping. This is the chain rule, tested numerically for every
forward pass diagnostic.

### M-R2: Capacity-normalized range

`0<=u_between<=1` and `0<=u_within<=1` whenever the denominator is positive;
the defined zero branch preserves the same range when all positive-mass roles
are singletons. This follows from the finite-support entropy bounds above.

### M-R3: Monotone direct-expert allocation

With all terms other than `u_between` fixed,

`partial g_M / partial u_between = g_M(1-g_M) softplus(beta) >= 0`.

Therefore M-RPG cannot reduce direct-role-expert weight as between-role
uncertainty rises. A free RPG MLP has no corresponding structural guarantee.

### M-R4: Convex evidence envelope

Since `g_M` lies in `[0,1]`, each coordinate `p_f(r)` lies between `p_d(r)`
and `p_m(r)`, and `p_f` remains on the four-role probability simplex. This
limits the gate to evidence weighting; it cannot synthesize a probability
outside both experts' support.

## Controlled Experiment Matrix

The existing 12-run RPG matrix remains the first controlled ablation:
RSG, RPG, no-within, no-between, and total-entropy-only, each for seeds
`20260727`, `20260728`, and `20260729`.

After it completes, M-RPG adds three registered configurations with the same
split, augmentation, EfficientNet-B0 backbone, optimizer, RSG regret flags,
residual verifiers, and seeds:

1. `mrpg_complete`: normalized `u_between`, normalized `u_within`, and
   non-negative `softplus(beta)` coefficient.
2. `mrpg_unconstrained_between`: identical inputs but a signed learned
   coefficient in place of `softplus(beta)`; this removes only M-R3.
3. `mrpg_without_between`: identical normalized base gate with the between-role
   term fixed to zero; this removes only role-relevant monotone routing.

The primary empirical comparison is M-RPG versus the completed RPG and RSG
baselines. The two M-RPG ablations identify whether any gain comes from the
monotone structural constraint rather than merely adding another scalar.

Metrics are Accuracy, Macro F1, target recall, target miss rate, two
hard-negative-to-target intrusion rates, Brier score, ECE, and routing regret.
Every performance statement must use the fixed three seeds and group-aware
paired Bootstrap intervals. A confidence interval crossing zero is reported as
no stable observed improvement.

## Novelty and Claim Boundary

Mixture-of-experts routing, entropy regularization, hierarchical labels, and
monotone neural networks are established techniques. The paper must not claim
to introduce any of them generically. The defensible claim is narrower:

"For a frozen mineral-species-to-beneficiation-role partition, we introduce a
candidate-capacity-normalized conditional-entropy gate whose direct-expert
allocation is provably nondecreasing in role-relevant species uncertainty."

The model is a visual recognition method under the public-specimen protocol.
No result may be described as industrial separation, mineral grade prediction,
recovery improvement, chemical identification, unknown-mineral rejection, or
stage-conditioned decision validation.

## Acceptance Criteria

1. Unit tests prove M-R1 numerically, M-R2 ranges, M-R3 derivative sign via
   finite differences and positive parameterization, and M-R4 simplex/envelope
   properties.
2. Existing RSG, CGDC, and RPG outputs remain unchanged unless their dedicated
   architecture flag is enabled.
3. A CPU smoke run exports the normalized uncertainties, gate, and model mode.
4. Each M-RPG configuration has three completed fixed-seed runs.
5. Report and paper include formulas, a structural figure, paired statistics,
   and the stated claim boundary; they describe an empirical gain only when
   its paired interval supports it.
