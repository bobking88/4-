"""Per-row implementation certificates; none is a recall or accuracy guarantee."""
from __future__ import annotations

import math

import torch

from lc_role_adapter import RoleResidualModel, anchor_geometry

ATOL = RTOL = 1e-10


def replay_without_adaptation(model: RoleResidualModel, batch: dict) -> dict:
    geometry = anchor_geometry(batch["log_anchor"])
    with torch.no_grad():
        return model.readout(batch["h"], batch["e"], batch["log_anchor"], geometry["b"])


def check_flat_reconstruction(result: dict, log_anchor: torch.Tensor) -> dict:
    geometry = anchor_geometry(log_anchor)
    v = result["v"]
    d = torch.cat([result["u"] + torch.logsumexp(geometry["logw"] + v, 1, keepdim=True), v], 1)
    flat = torch.log_softmax(log_anchor + d, 1)
    residual = (flat-result["logq"]).abs().max(1).values
    if not torch.isfinite(residual).all():
        raise ValueError("Nonfinite flat reconstruction diagnostics.")
    return dict(logq=flat, residuals=residual.detach(),
                max_abs_residual=float(residual.max()),
                violation_count=int((residual > ATOL).sum()))


def audit_theory(model: RoleResidualModel, batch: dict, result: dict) -> dict:
    with torch.no_grad():
        return _audit(model, batch, result)


def _audit(model: RoleResidualModel, batch: dict, result: dict) -> dict:
    anchor, logq = batch["log_anchor"], result["logq"]
    geometry = anchor_geometry(anchor)
    n = len(anchor)
    if (batch["labels"].shape != (n,) or len(batch["records"]) != n
            or bool(((batch["labels"] < 0) | (batch["labels"] > 3)).any())
            or any(not torch.isfinite(value).all() for value in result.values() if torch.is_tensor(value))):
        raise ValueError("Invalid aligned audit inputs.")
    bounded = model.arm != "U1"
    b = geometry["b"][:, 0]
    delta_norms = result["delta"].norm(dim=2)
    adaptation = (result["adapted_h"]-batch["h"]).norm(dim=1)
    flat = check_flat_reconstruction(result, anchor)
    plain = replay_without_adaptation(model, batch)
    log_change = logq - plain["logq"]
    logq_drift = log_change.abs().max(1).values
    odds_drift = log_change.max(1).values - log_change.min(1).values
    winners = plain["logq"].argmax(1)
    sorted_plain = plain["logq"].sort(1, descending=True).values
    plain_margin = sorted_plain[:, 0] - sorted_plain[:, 1]
    changed = winners != logq.argmax(1)
    L_F = (1+math.exp(-1))*float(torch.linalg.matrix_norm(model.head[2].weight, 2))*float(torch.linalg.matrix_norm(model.head[0].weight[:, :64], 2))
    alpha, beta = model.alpha_max*b, model.beta_max*b
    eta = (model.alpha_max+model.beta_max)*L_F*b.square()
    protected = (plain_margin > eta) if bounded else torch.zeros(n, dtype=torch.bool)
    wmax = geometry["logw"].max(1, keepdim=True).values
    log_rho = wmax[:, 0] - torch.logsumexp(torch.minimum(wmax, geometry["logw"]+beta[:, None]), 1)
    envelope = geometry["logit_t"][:, 0] + alpha - log_rho
    target_margin = logq[:, 0] - logq[:, 1:].max(1).values
    losses_drift = (logq-anchor).abs()
    drift_bounds = torch.cat([alpha[:, None], (alpha+beta)[:, None].expand(-1, 3)], 1)
    kl = (anchor.exp()*(anchor-logq)).sum(1)
    kl_bound = (alpha.square() + anchor[:, 1:].exp().sum(1)*beta.square())/8
    zeros = torch.zeros(n, dtype=torch.float64)
    ones = torch.ones_like(zeros)
    expected_g = (geometry["b"].expand(-1, 3)/3 if model.arm in ("S0", "S1") else
                  geometry["g"] if len(model.adapters) else torch.zeros(n, 3, dtype=torch.float64))
    composition = batch["h"]+(result["g"][:, :, None]*result["delta"]).sum(1)
    # Each tuple is an actual value and its upper bound, never an averaged gate.
    checks = {
        "delta_norm": (delta_norms.max(1).values, ones),
        "adaptation_norm": (adaptation, b),
        "routing_reconstruction": ((result["g"]-expected_g).abs().max(1).values, zeros),
        "feature_reconstruction": ((result["adapted_h"]-composition).abs().max(1).values, zeros),
        "b_reconstruction": ((result["b"][:, 0]-b).abs(), zeros),
        "probability_sum": (torch.logsumexp(logq, 1).abs(), zeros),
        "positive_probability": ((logq.exp() == 0).any(1).double(), zeros),
        "flat_reconstruction": (flat["residuals"], zeros),
    }
    if bounded:
        checks.update({
            "alpha_reconstruction": ((result["alpha"][:, 0]-alpha).abs(), zeros),
            "beta_reconstruction": ((result["beta"][:, 0]-beta).abs(), zeros),
            "u_bound": (result["u"][:, 0].abs(), alpha),
            "v_span": (result["v"].max(1).values-result["v"].min(1).values, beta),
            "logq_drift": (logq_drift, eta),
            "pair_log_odds_drift": (odds_drift, eta),
            "protected_winner_changed": ((changed & protected).double(), zeros),
            "target_margin_envelope": (target_margin, envelope),
            "nll_drift": ((losses_drift-drift_bounds).max(1).values, zeros),
            "kl_drift": (kl, kl_bound),
        })
    violations = torch.zeros(n, dtype=torch.bool)
    violated_by = [[] for _ in range(n)]
    maxima = {}
    for name, (actual, bound) in checks.items():
        residual = actual-bound
        fail = residual > ATOL+RTOL*bound.abs()
        violations |= fail
        maxima[name] = float(residual.max())
        for index in fail.nonzero().flatten().tolist():
            violated_by[index].append(name)
    rows = []
    for i, record in enumerate(batch["records"]):
        y = int(batch["labels"][i])
        rows.append(dict(image_id=record["image_id"], class_id=y,
                         delta_norms=delta_norms[i].tolist(), adaptation_norm=float(adaptation[i]),
                         b=float(b[i]), alpha=float(alpha[i]), beta=float(beta[i]),
                         abs_u=float(result["u"][i].abs()), v_span=float(result["v"][i].max()-result["v"][i].min()),
                         L_F=L_F, eta=float(eta[i]) if bounded else None,
                         logq_drift=float(logq_drift[i]), pair_log_odds_drift=float(odds_drift[i]),
                         plain_margin=float(plain_margin[i]), winner_protected=bool(protected[i]),
                         winner_changed=bool(changed[i]),
                         target_margin_envelope=float(envelope[i]) if bounded else None,
                         actual_target_margin=float(target_margin[i]),
                         target_impossible=bool(envelope[i] < 0) if bounded else None,
                         nll_drift=float(losses_drift[i, y]), nll_drift_bound=float(drift_bounds[i, y]) if bounded else None,
                         kl=float(kl[i]), kl_bound=float(kl_bound[i]) if bounded else None,
                         violations=violated_by[i]))
    protected_count = int(protected.sum())
    return dict(rows=rows, row_count=n, violation_count=int(violations.sum()),
                violating_image_ids=[rows[i]["image_id"] for i in violations.nonzero().flatten().tolist()],
                violating_rows=[rows[i] for i in violations.nonzero().flatten().tolist()],
                max_residual_by_check=maxima, L_F=L_F,
                maximum_absolute_flat_residual=flat["max_abs_residual"],
                protected_winner_count=protected_count, protected_winner_fraction=protected_count/n,
                loose_certificate_fraction=float((eta >= plain_margin).double().mean()) if bounded else None,
                target_impossible_count=int((envelope < 0).sum()) if bounded else None,
                eta_min=float(eta.min()) if bounded else None, eta_max=float(eta.max()) if bounded else None,
                eta_mean=float(eta.mean()) if bounded else None,
                bounded_probability_certificate_applicable=bounded,
                auxiliary_b_squared_certificate_applicable=False,
                certificate_scope="same-weights main-path perturbation, not accuracy or recall")
