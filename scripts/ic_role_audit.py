"""Conditional IC-RFA implementation certificates, never accuracy guarantees."""
from __future__ import annotations

import math

import torch

from ic_role_model import ICRoleModel, _digest, reference_identity
from lc_role_adapter import anchor_geometry, bounded_log_probabilities
from lc_role_audit import check_flat_reconstruction

ATOL = RTOL = 1e-10
FD_STEP = 1e-6


def _references(model: ICRoleModel) -> None:
    identity = reference_identity(model)
    if not all(identity[k] for k in ("references_unaliased", "references_frozen", "references_match_initial")):
        raise ValueError("Function-reference or regularization-origin identity changed.")


def _forward(model, batch, **kwargs):
    return model(batch["h"], batch["e"], batch["log_anchor"], **kwargs)


def _role(role):
    if type(role) is not int or role not in (1, 2, 3):
        raise ValueError("Expected negative role class ID 1, 2 or 3.")
    return role-1


def _scalar(model, batch, role, view):
    result = _forward(model, batch, auxiliary=True)
    logq = result["logq"] if view == "main" else result["aux_logq"][:, role-1]
    return logq[:, 0]-logq[:, role]


def scalar_branch_jacobian(model: ICRoleModel, batch: dict, role: int, view: str) -> dict:
    index = _role(role)
    if view not in ("main", "aux"):
        raise ValueError("Expected main or aux scalar view.")
    _references(model)
    if not model.adapters:
        return dict(status="NOT_APPLICABLE", jacobian=None, layout=[], reason="no_adapter")
    named = list(model.adapters[index].named_parameters())
    if any(not p.requires_grad for _, p in named):
        raise ValueError("Branch Jacobian requires trainable branch parameters.")
    layout, offset = [], 0
    for name, parameter in named:
        layout.append(dict(name=f"adapters.{index}.{name}", shape=list(parameter.shape),
                           offset=offset, numel=parameter.numel()))
        offset += parameter.numel()
    values = _scalar(model, batch, role, view)
    rows = []
    for i in range(len(values)):
        gradients = torch.autograd.grad(values[i], tuple(p for _, p in named),
                                        retain_graph=i+1 < len(values), allow_unused=True)
        rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                               for (_, p), g in zip(named, gradients, strict=True)]).detach())
    matrix = torch.stack(rows)
    if not torch.isfinite(matrix).all():
        raise ValueError("Nonfinite scalar branch Jacobian.")
    return dict(status="COMPLETE", jacobian=matrix, layout=layout, role=role, view=view,
                row_ids=[r["image_id"] for r in batch["records"]])


def audit_directional_derivative(model: ICRoleModel, batch: dict, role: int,
                                 parameter: str, direction: torch.Tensor) -> dict:
    index = _role(role)
    _references(model)
    if not model.adapters or parameter not in dict(model.adapters[index].named_parameters()):
        raise ValueError("Unknown branch parameter for directional derivative.")
    value = dict(model.adapters[index].named_parameters())[parameter]
    if (not torch.is_tensor(direction) or direction.shape != value.shape
            or direction.dtype != torch.float64 or direction.device.type != "cpu"
            or not torch.isfinite(direction).all()):
        raise ValueError("Expected finite aligned float64 direction.")
    jac = scalar_branch_jacobian(model, batch, role, "aux")
    block = next(r for r in jac["layout"] if r["name"] == f"adapters.{index}.{parameter}")
    analytic = jac["jacobian"][:, block["offset"]:block["offset"]+block["numel"]] @ direction.flatten()
    original = value.detach().clone()
    try:
        with torch.no_grad():
            value.copy_(original+FD_STEP*direction)
            plus = _scalar(model, batch, role, "aux").detach().clone()
            value.copy_(original-FD_STEP*direction)
            minus = _scalar(model, batch, role, "aux").detach().clone()
    finally:
        with torch.no_grad():
            value.copy_(original)
    finite_difference = (plus-minus)/(2*FD_STEP)
    if not torch.isfinite(finite_difference).all():
        raise ValueError("Nonfinite fixed-step finite difference.")
    error = (analytic-finite_difference).abs()
    tolerance = 1e-7+1e-5*analytic.abs()
    return dict(analytic=analytic, finite_difference=finite_difference, step=FD_STEP,
                passed=bool((error <= tolerance).all()), max_abs_error=float(error.max()),
                role=role, parameter=parameter, scope="aux scalar branch parameter derivative")


def audit_initial_kernel(model: ICRoleModel, batch: dict, role: int) -> dict:
    _role(role)
    _references(model)
    if not model.adapters:
        return dict(status="NOT_APPLICABLE", reason="no_adapter")
    if _digest(model.named_current()) != model.initial_current_sha256:
        raise ValueError("Initial-kernel identity cannot be asserted on a trained state.")
    main = scalar_branch_jacobian(model, batch, role, "main")
    auxiliary = scalar_branch_jacobian(model, batch, role, "aux")
    g = _forward(model, batch)["g"][:, role-1].detach()
    G = torch.diag(g)
    jm, ja = main["jacobian"], auxiliary["jacobian"]
    km, ka = jm @ jm.T, ja @ ja.T
    jac_error = (jm-g[:, None]*ja).abs().max()
    gram_error = (km-G @ ka @ G).abs().max()
    eigen_aux = torch.linalg.eigvalsh(ka)
    eigen_main = torch.linalg.eigvalsh(km)
    aux_rank = int((eigen_aux > 1e-10).sum())
    main_rank = int((eigen_main > 1e-10).sum())
    numeric_pd = bool(eigen_aux.min() > 1e-10)
    positive_gates = bool((g > 0).all())
    lower_bound = float(g.min().square()*eigen_aux.min()) if numeric_pd and positive_gates else None
    return dict(status="COMPLETE", passed=bool(jac_error <= ATOL and gram_error <= ATOL),
                Jmain=jm, Jaux=ja, Kmain=km, Kaux=ka, G=G,
                max_jacobian_residual=float(jac_error), max_gram_residual=float(gram_error),
                aux_eigenvalues=eigen_aux, main_eigenvalues=eigen_main,
                aux_numeric_rank=aux_rank, main_numeric_rank=main_rank,
                aux_numeric_positive_definite=numeric_pd,
                positive_gates=positive_gates, numeric_rank_threshold=1e-10,
                conditional_lambda_lower_bound=lower_bound,
                row_ids=main["row_ids"], layout=main["layout"],
                scope="initial scalar branch block only; numeric PD is not a proof or convergence claim")


@torch.no_grad()
def audit_bounds(model: ICRoleModel, batch: dict, result: dict) -> dict:
    _references(model)
    anchor, logq = batch["log_anchor"], result["logq"]
    geometry = anchor_geometry(anchor)
    n = len(anchor)
    labels, records = batch["labels"], batch["records"]
    if (labels.shape != (n,) or labels.dtype != torch.long or labels.device.type != "cpu"
            or len(records) != n or bool(((labels < 0) | (labels > 3)).any())
            or logq.shape != anchor.shape
            or any(not torch.isfinite(v).all() for v in result.values() if torch.is_tensor(v))):
        raise ValueError("Invalid aligned certificate inputs.")
    b = geometry["b"][:, 0]
    alpha = beta = .5*b
    L_F = (1+math.exp(-1))*float(torch.linalg.matrix_norm(model.head[2].weight, 2))*float(
        torch.linalg.matrix_norm(model.head[0].weight[:, :64], 2))
    eta_main, eta_aux = L_F*b.square(), L_F*b
    off = _forward(model, batch, adaptation=False)["logq"]
    change = logq-off
    single_drift = change.abs().max(1).values
    pair_drift = change.max(1).values-change.min(1).values
    ordered = off.sort(1, descending=True).values
    margin = ordered[:, 0]-ordered[:, 1]
    protected = margin > eta_main
    winner_changed = logq.argmax(1) != off.argmax(1)
    delta_norm = result["delta"].norm(dim=2)
    feature_norm = (result["adapted_h"]-batch["h"]).norm(dim=1)
    expected_g = (b[:, None].expand(-1, 3)/3 if model.configuration["routing"] == "U"
                  else geometry["g"] if model.adapters else torch.zeros_like(geometry["g"]))
    composition = (batch["h"]+(result["g"][:, :, None]*result["delta"]).sum(1)
                   if result.get("adaptation_enabled", True) else batch["h"])
    flat = check_flat_reconstruction(result, anchor)
    wmax = geometry["logw"].max(1, keepdim=True).values
    log_rho = wmax[:, 0]-torch.logsumexp(torch.minimum(wmax, geometry["logw"]+beta[:, None]), 1)
    envelope = geometry["logit_t"][:, 0]+alpha-log_rho
    target_margin = logq[:, 0]-logq[:, 1:].max(1).values
    drift_bound = torch.cat([alpha[:, None], (alpha+beta)[:, None].expand(-1, 3)], 1)
    drift = (logq-anchor).abs()
    anchor_kl = (anchor.exp()*(anchor-logq)).sum(1)
    anchor_kl_bound = (alpha.square()+anchor[:, 1:].exp().sum(1)*beta.square())/8
    model_kl = (logq.exp()*(logq-off)).sum(1)
    model_kl_bound = (alpha.square()+logq[:, 1:].exp().sum(1)*beta.square())/2
    zero = torch.zeros_like(b)
    reference_score = (model.head_ref(torch.cat([batch["h"], batch["e"]], 1))
                       if model.head_ref is not None else torch.zeros_like(logq))

    def defined_readout(visual):
        scores = model.head(torch.cat([visual, batch["e"]], 1))-reference_score
        defined = bounded_log_probabilities(anchor, scores, geometry["b"], alpha_max=.5, beta_max=.5)
        return scores, defined["logq"]

    expected_scores, expected_main = defined_readout(result["adapted_h"])
    checks = {
        "delta_norm": (delta_norm.max(1).values, torch.ones_like(b)),
        "feature_norm": (feature_norm, b),
        "feature_composition": ((result["adapted_h"]-composition).abs().max(1).values, zero),
        "routing": ((result["g"]-expected_g).abs().max(1).values, zero),
        "budget": ((result["b"][:, 0]-b).abs(), zero),
        "alpha": ((result["alpha"][:, 0]-alpha).abs(), zero),
        "beta": ((result["beta"][:, 0]-beta).abs(), zero),
        "normalization": (torch.logsumexp(logq, 1).abs(), zero),
        "positive_probability": ((logq.exp() == 0).any(1).double(), zero),
        "flat_reconstruction": (flat["residuals"], zero),
        "main_score_definition": ((result["scores"]-expected_scores).abs().max(1).values, zero),
        "main_probability_definition": ((logq-expected_main).abs().max(1).values, zero),
        "u_bound": (result["u"][:, 0].abs(), alpha),
        "v_span": (result["v"].max(1).values-result["v"].min(1).values, beta),
        "main_logq_drift": (single_drift, eta_main),
        "main_pair_drift": (pair_drift, eta_main),
        "protected_winner": ((protected & winner_changed).double(), zero),
        "target_margin_envelope": (target_margin, envelope),
        "nll_drift": ((drift-drift_bound).max(1).values, zero),
        "anchor_kl": (anchor_kl, anchor_kl_bound),
        "model_pair_kl": (model_kl, model_kl_bound),
    }
    auxiliary_row_count = 0
    auxiliary_max_drift = torch.zeros_like(b)
    if "aux_logq" in result:
        auxiliary_logq = result["aux_logq"]
        if auxiliary_logq.shape != (n, 3, 4):
            raise ValueError("Invalid auxiliary certificate shape.")
        auxiliary_row_count = n*3
        auxiliary_change = auxiliary_logq-off[:, None, :]
        auxiliary_max_drift = auxiliary_change.abs().amax((1, 2))
        auxiliary_pair = (auxiliary_change.max(2).values-auxiliary_change.min(2).values).max(1).values
        defined_auxiliary = torch.stack([defined_readout(batch["h"]+result["delta"][:, j])[1]
                                         for j in range(3)], 1)
        aux_nll_residual = ((auxiliary_logq-anchor[:, None, :]).abs()-drift_bound[:, None, :]).amax((1, 2))
        aux_kl = (anchor.exp()[:, None, :]*(anchor[:, None, :]-auxiliary_logq)).sum(2).max(1).values
        aux_margin = (auxiliary_logq[:, :, 0]-auxiliary_logq[:, :, 1:].max(2).values).max(1).values
        checks["aux_logq_drift"] = (auxiliary_max_drift, eta_aux)
        checks["aux_pair_drift"] = (auxiliary_pair, eta_aux)
        checks["aux_definition"] = ((auxiliary_logq-defined_auxiliary).abs().amax((1, 2)), zero)
        checks["aux_nll_drift"] = (aux_nll_residual, zero)
        checks["aux_anchor_kl"] = (aux_kl, anchor_kl_bound)
        checks["aux_target_margin_envelope"] = (aux_margin, envelope)
        checks["aux_normalization"] = (torch.logsumexp(auxiliary_logq, 2).abs().max(1).values, zero)
        checks["aux_positive_probability"] = ((auxiliary_logq.exp() == 0).any(2).any(1).double(), zero)
    failed = torch.zeros(n, dtype=torch.bool)
    reasons = [[] for _ in range(n)]
    maxima = {}
    for name, (actual, bound) in checks.items():
        residual = actual-bound
        if not torch.isfinite(residual).all():
            raise ValueError("Nonfinite certificate residual.")
        violation = residual > ATOL+RTOL*bound.abs()
        failed |= violation
        maxima[name] = float(residual.max())
        for i in violation.nonzero().flatten().tolist():
            reasons[i].append(name)
    rows = [dict(image_id=record["image_id"], class_id=int(labels[i]), b=float(b[i]), L_F=L_F,
                 delta_norms=delta_norm[i].tolist(), feature_norm=float(feature_norm[i]),
                 eta_main=float(eta_main[i]), eta_aux=float(eta_aux[i]),
                 actual_main_logq_drift=float(single_drift[i]), actual_pair_drift=float(pair_drift[i]),
                 actual_aux_logq_drift=float(auxiliary_max_drift[i]), off_margin=float(margin[i]),
                 winner_protected=bool(protected[i]), winner_changed=bool(winner_changed[i]),
                 target_margin_envelope=float(envelope[i]), target_margin=float(target_margin[i]),
                 nll_drift=float(drift[i, int(labels[i])]), anchor_kl=float(anchor_kl[i]),
                 anchor_kl_bound=float(anchor_kl_bound[i]), model_kl=float(model_kl[i]),
                 model_kl_bound=float(model_kl_bound[i]), violations=reasons[i])
            for i, record in enumerate(records)]
    return dict(rows=rows, row_count=n, auxiliary_row_count=auxiliary_row_count,
                violation_count=int(failed.sum()),
                violating_image_ids=[records[i]["image_id"] for i in failed.nonzero().flatten().tolist()],
                max_residual_by_check=maxima, L_F=L_F,
                protected_winner_count=int(protected.sum()), protected_winner_fraction=float(protected.double().mean()),
                loose_certificate_fraction=float((eta_main >= margin).double().mean()),
                off_anchor_max_log_difference=float((off-anchor).abs().max()),
                scope="same-weight perturbation and bounded probability geometry, not label correctness")
