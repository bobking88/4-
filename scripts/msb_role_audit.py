"""Numerical contracts and synthetic-only checks of conditional properties."""
from __future__ import annotations

import copy
import itertools
import math
import torch
from torch.nn import functional as F
from msb_role_controls import control_forward, convex_objective, objective
from msb_role_model import NEURAL_UNITS, ROLE_UNITS, anchor_margins, unit_spec


def _close(actual, expected, *, atol=1e-10, rtol=1e-10):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError("MSB numerical contract mismatch.")
    return float((actual - expected).detach().abs().max())


def audit_forward_contract(batch, result) -> dict:
    logp = batch["log_anchor"]
    anchor_margins(logp)
    m = logp[:, :1] - logp[:, 1:]
    radius = 2 * m.abs() + .25
    unit_spec(result["unit"])
    q = result["logq"]
    if (q.shape != logp.shape or q.dtype != torch.float64 or q.device.type != "cpu"
            or not torch.isfinite(q).all() or bool((q.exp() == 0).any())):
        raise ValueError("NUMERIC_RANGE_FAILURE: invalid audited probabilities")
    _close(torch.logsumexp(q, 1), torch.zeros(len(q), dtype=torch.float64))
    _close(result["m"], m)
    _close(result["R_ref"], radius)
    error = _close(q[:, :1] - q[:, 1:] - m, result["d"])
    _close(torch.log_softmax(result["scores"], 1), q)
    bounded = result["unit"] in NEURAL_UNITS and result["unit"] != "UNB-SCALE"
    if result["unit"] == "BIAS3-CVX": bounded = True
    bound = torch.ones_like(radius) if result["unit"] == "FIX1" else radius
    if bounded:
        tolerance = 1e-10 + 1e-10 * bound.abs()
        if bool((result["d"].abs() > bound + tolerance).any()):
            raise ValueError("MSB closed output budget violated.")
        a = bound.max(1).values
        drift = (q - logp).abs()
        if (bool((drift[:, 0] > a + 1e-10 + 1e-10*a).any())
                or bool((drift[:, 1:] > bound + a[:, None] + 1e-10 + 1e-10*(bound+a[:,None])).any())):
            raise ValueError("MSB log-probability drift bound violated.")
    c = result.get("c")
    saturation = int((c.abs() == 1).sum()) if torch.is_tensor(c) else 0
    if saturation != result.get("saturation_count", 0):
        raise ValueError("Tanh saturation metadata mismatch.")
    return dict(max_root_odds_error=error, budget_checked=bounded,
                saturation_count=saturation, has_saturation=saturation > 0)


def gradient_components(model, result, labels) -> dict:
    terms = objective(model, result, labels)
    named = model.named_current()
    parameters = tuple(p for _, p in named)
    vectors = {}
    for key in ("nll", "pair_weighted", "relative_penalty", "sp", "data", "regularization", "total"):
        gradients = torch.autograd.grad(terms[key], parameters, retain_graph=True, allow_unused=True)
        vectors[key] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                  for p, g in zip(parameters, gradients)])
        if not torch.isfinite(vectors[key]).all():
            raise ValueError("NUMERIC_RANGE_FAILURE: component gradient nonfinite")
    _close(vectors["data"], vectors["nll"] + vectors["pair_weighted"])
    _close(vectors["regularization"], vectors["relative_penalty"] + vectors["sp"])
    _close(vectors["total"], vectors["data"] + vectors["regularization"])
    layout, offset = [], 0
    for name, parameter in named:
        layout.append(dict(name=name, shape=list(parameter.shape), offset=offset, numel=parameter.numel()))
        offset += parameter.numel()
    return dict(vectors=vectors, layout=layout,
                norms={key: float(value.norm()) for key, value in vectors.items()},
                losses={key: float(value.detach()) for key, value in terms.items()})


def audit_control_equivalence(main, control, batch) -> dict:
    result = main(batch["h"], batch["e"], batch["log_anchor"])
    if not isinstance(control, dict) or control.get("unit") != "BIAS3-CVX" or main.unit != "MAIN":
        raise ValueError("This objective equivalence audit requires the bias-only MAIN submodel.")
    for name, p in main.named_current():
        if not name.endswith("zeta") and not torch.equal(p.detach(), main.initial_parameter(name)):
            raise ValueError("MAIN has non-bias updates; bias-only equivalence does not apply.")
    other = control_forward(control, batch["h"], batch["e"], batch["log_anchor"])
    error = _close(result["logq"], other["logq"], atol=1e-12, rtol=1e-12)
    loss = objective(main, result, batch["labels"])["total"]
    other_loss = convex_objective(control, control["coordinates"], batch)
    objective_error = _close(loss, other_loss, atol=1e-12, rtol=1e-12)
    return dict(max_logq_error=error, objective_error=objective_error)


def _origin(model):
    clone = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in clone.named_current(): p.copy_(clone.initial_parameter(name))
    return clone


def _set_bias(model, values):
    with torch.no_grad():
        for branch, value in zip(model.branches, values):
            branch.zeta.copy_(branch.zeta0 + value)


def _grid_batch(margins):
    logits = torch.cat([torch.zeros(len(margins), 1, dtype=torch.float64), -margins], 1)
    return dict(h=torch.zeros(len(margins), 64, dtype=torch.float64),
                e=torch.zeros(len(margins), 23, dtype=torch.float64),
                log_anchor=torch.log_softmax(logits, 1))


def _forward(model, batch):
    return model(batch["h"], batch["e"], batch["log_anchor"])


def audit_synthetic_theory(model, batch) -> dict:
    if (model.unit != "MAIN" or not batch.get("records")
            or any(not r.get("image_id", "").startswith("synthetic-") for r in batch["records"])):
        raise ValueError("Synthetic-only origin theorem checks; no real data accepted.")
    origin = _origin(model)
    values = (-50., -10., 0., 10., 50.)
    margins = torch.tensor(list(itertools.product(values, repeat=3)), dtype=torch.float64)
    grid = _grid_batch(margins)
    witness = math.atanh(.5)
    winner_min, negative_min = float("inf"), float("inf")
    witness_count, energy_sum, sp_max = 0, 0., 0.
    for target in range(4):
        candidate = _origin(origin)
        biases = [witness] * 3
        if target: biases[target-1] = -witness
        _set_bias(candidate, biases)
        r = _forward(candidate, grid)
        audit_forward_contract(grid, r)
        energy = (r["d"] / r["R_ref"]).square().mean(1)
        _close(energy, torch.full_like(energy, .25), atol=1e-12, rtol=1e-12)
        witness_count += len(energy)
        energy_sum += float(energy.detach().sum())
        sp_distance = sum((p-candidate.initial_parameter(name)).detach().square().sum()
                          for name,p in candidate.named_current())
        sp_max = max(sp_max, float(sp_distance))
        if not bool((r["logq"].argmax(1) == target).all()):
            raise ValueError("Interior witness does not make its class win.")
        rival = [k for k in range(4) if k != target]
        delta = r["logq"][:, target:target+1] - r["logq"][:, rival]
        winner_min = min(winner_min, float(delta.detach().min()))
        if target:
            negative_rival = [k for k in range(1, 4) if k != target]
            negative_min = min(negative_min, float((r["logq"][:,target:target+1]-r["logq"][:,negative_rival]).detach().min()))
    if winner_min < .125-1e-10 or negative_min < .25-1e-10:
        raise ValueError("Witness lower margin bounds failed.")

    boundary_derivatives, loss_derivatives, initial_gradients, fd_errors = [], [], [], []
    for value in values:
        local = _grid_batch(torch.full((1,3), value, dtype=torch.float64))
        candidate = _origin(origin)
        radius = 2*abs(value)+.25
        s = math.atanh(-value/radius)
        _set_bias(candidate, [s, 0., 0.])
        r = _forward(candidate, local)
        margin = r["logq"][0,0]-r["logq"][0,1]
        _close(margin, torch.zeros((), dtype=torch.float64))
        derivative = torch.autograd.grad(margin, candidate.branches[0].zeta, retain_graph=True)[0]
        loss_derivative = torch.autograd.grad(F.softplus(-margin), candidate.branches[0].zeta)[0]
        boundary_derivatives.append(float(derivative.detach()))
        loss_derivatives.append(float(loss_derivative.detach().abs()))
        epsilon = 1e-6
        measurements = []
        for displacement in (epsilon, -epsilon):
            _set_bias(candidate, [s+displacement,0.,0.])
            rr = _forward(candidate, local)
            measurements.append(float((rr["logq"][0,0]-rr["logq"][0,1]).detach()))
        fd = (measurements[0]-measurements[1])/(2*epsilon)
        error = abs(fd-float(derivative.detach()))
        if error > 1e-7+1e-5*abs(float(derivative.detach())):
            raise ValueError("Boundary finite difference disagrees with autograd.")
        fd_errors.append(error)
        candidate = _origin(origin)
        r = _forward(candidate, local)
        m = r["logq"][0,0]-r["logq"][0,1]
        sign = -1. if value > 0 else 1.
        g = torch.autograd.grad(F.softplus(-sign*m), candidate.branches[0].zeta)[0]
        initial_gradients.append(float(g.detach().abs()))
    if (min(boundary_derivatives)<.1875-1e-10 or min(loss_derivatives)<.09375-1e-10
            or min(initial_gradients)<.125-1e-10):
        raise ValueError("Conditional derivative lower bounds failed.")

    candidate = _origin(origin)
    local = _grid_batch(torch.tensor([[-10.,0.,10.]], dtype=torch.float64))
    _set_bias(candidate, [witness]*3)
    r = _forward(candidate, local)
    normalized = (r["logq"][:,:1]-r["logq"][:,1:])/r["R_ref"]
    jacobian = torch.stack([torch.stack(torch.autograd.grad(normalized[0,j], [b.zeta for b in candidate.branches],retain_graph=True))
                            for j in range(3)])
    _close(jacobian, .75*torch.eye(3,dtype=torch.float64))

    branch = copy.deepcopy(origin.branches[0])
    with torch.no_grad(): branch.U[0,0].add_(.2)
    h, e = torch.zeros(64,dtype=torch.float64), torch.zeros(23,dtype=torch.float64)
    function = lambda x,y:(branch.score(x[None],y[None])-branch.fixed_score(x[None],y[None])).sum()
    cross = torch.autograd.functional.hessian(function,(h,e),vectorize=True)[0][1]
    analytic = ((branch.U.T * branch.a) @ branch.V - (branch.U0.T * branch.a0) @ branch.V0)/2
    cross_error = _close(cross, analytic)
    cross_norm = float(cross.detach().norm())
    if cross_norm <= 1e-8: raise ValueError("Constructed mixed Hessian witness degenerated.")

    same = _grid_batch(torch.zeros(2,3,dtype=torch.float64))
    predictions = _forward(origin,same)["logq"].argmax(1)
    conflicting_correct = int((predictions==torch.tensor([0,1])).sum())
    if conflicting_correct>1: raise ValueError("Deterministic identical-input counterexample failed.")
    high = _grid_batch(torch.full((1,3),50.,dtype=torch.float64))
    before = int(_forward(origin,high)["logq"].argmax(1)[0])
    candidate = _origin(origin); _set_bias(candidate,[-witness,witness,witness])
    after = int(_forward(candidate,high)["logq"].argmax(1)[0])
    if before!=0 or after==0: raise ValueError("High-confidence harm counterexample failed.")
    other = _grid_batch(torch.tensor([[0.,-3.,0.]],dtype=torch.float64))
    other_q = _forward(origin,other)["logq"]
    nonwinning = bool(abs(float((other_q[0,0]-other_q[0,1]).detach()))<1e-10 and int(other_q.argmax(1)[0])==2)
    return dict(scope="SYNTHETIC_CONDITIONAL_PROPERTIES_NOT_EFFECT", interior_witness_count=witness_count,
                min_witness_winner_margin=winner_min, min_witness_negative_margin=negative_min,
                witness_relative_energy=energy_sum/witness_count, witness_sp_squared_distance_max=sp_max,
                min_boundary_bias_derivative=min(boundary_derivatives),
                min_boundary_pair_loss_derivative=min(loss_derivatives),
                min_wrong_pair_initial_bias_gradient=min(initial_gradients),
                max_finite_difference_error=max(fd_errors), normalized_bias_jacobian=jacobian.detach().tolist(),
                mixed_hessian_norm=cross_norm, mixed_hessian_analytic_error=cross_error,
                conflicting_input_correct_count=conflicting_correct, correct_anchor_label=before,
                harmful_witness_label=after, pair_boundary_can_be_nonwinning=nonwinning,
                joint_fit_guaranteed=False, global_convergence_proved=False,
                parameter_identifiability_proved=False, all_confident_correct_protected=False,
                non_equivalence_to_general_mlp_proved=False)
