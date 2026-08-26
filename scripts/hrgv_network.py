from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from train_role_aware_mineral_classifier import compute_role_aware_contrastive_loss


def _validate_probability_matrix(probabilities, name: str) -> None:
    if probabilities.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, classes].")
    if probabilities.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two classes.")


def normalized_entropy(probabilities, torch):
    """Return entropy normalized to [0, 1] for each batch row."""
    _validate_probability_matrix(probabilities, "probabilities")
    epsilon = torch.finfo(probabilities.dtype).eps
    safe = probabilities.clamp_min(epsilon)
    safe = safe / safe.sum(dim=1, keepdim=True).clamp_min(epsilon)
    entropy = -(safe * safe.log()).sum(dim=1, keepdim=True)
    return entropy / math.log(probabilities.shape[1])


def jensen_shannon_divergence(first, second, torch):
    """Return per-image Jensen-Shannon divergence between role posteriors."""
    if first.shape != second.shape:
        raise ValueError("Probability tensors must have the same shape.")
    _validate_probability_matrix(first, "first")
    epsilon = torch.finfo(first.dtype).eps
    first_safe = first.clamp_min(epsilon)
    second_safe = second.clamp_min(epsilon)
    first_safe = first_safe / first_safe.sum(dim=1, keepdim=True).clamp_min(epsilon)
    second_safe = second_safe / second_safe.sum(dim=1, keepdim=True).clamp_min(epsilon)
    midpoint = 0.5 * (first_safe + second_safe)
    first_kl = (first_safe * (first_safe.log() - midpoint.log())).sum(dim=1, keepdim=True)
    second_kl = (second_safe * (second_safe.log() - midpoint.log())).sum(dim=1, keepdim=True)
    return 0.5 * (first_kl + second_kl)


def adapter_decomposition_loss(direct_delta, species_delta, torch):
    """Penalize collinear expert residuals while preserving their magnitudes."""
    if direct_delta.shape != species_delta.shape:
        raise ValueError("Adapter residual tensors must have the same shape.")
    if direct_delta.ndim != 2:
        raise ValueError("Adapter residual tensors must have shape [batch, features].")
    cosine_similarity = functional.cosine_similarity(direct_delta, species_delta, dim=1)
    return cosine_similarity.square().mean()


def disagreement_calibrated_role_probabilities(
    fused,
    direct,
    mapped,
    bounded_residual,
    torch,
):
    """Calibrate a fused posterior only in proportion to expert disagreement."""
    if fused.shape != direct.shape or direct.shape != mapped.shape:
        raise ValueError("Fused, direct, and mapped posteriors must have matching shapes.")
    if bounded_residual.shape != fused.shape:
        raise ValueError("Calibration residual must have the same shape as role posteriors.")
    _validate_probability_matrix(fused, "fused")
    epsilon = torch.finfo(fused.dtype).eps
    disagreement_gain = 1.0 - torch.exp(
        -jensen_shannon_divergence(direct, mapped, torch)
    )
    bounded_residual = torch.tanh(bounded_residual)
    calibrated_logits = fused.clamp_min(epsilon).log() + disagreement_gain * bounded_residual
    return torch.softmax(calibrated_logits, dim=1), disagreement_gain


def mix_role_experts(direct, mapped, gate):
    """Mix direct and species-mapped role posteriors using a scalar image gate."""
    if direct.shape != mapped.shape:
        raise ValueError("Direct and mapped probabilities must have the same shape.")
    _validate_probability_matrix(direct, "direct")
    if gate.shape != (direct.shape[0], 1):
        raise ValueError("Gate must have shape [batch, 1].")
    if bool(((gate < 0) | (gate > 1)).any()):
        raise ValueError("Gate values must lie in [0, 1].")
    mixed = gate * direct + (1.0 - gate) * mapped
    epsilon = direct.new_tensor(torch.finfo(direct.dtype).eps)
    return mixed / mixed.sum(dim=1, keepdim=True).clamp_min(epsilon)


def regret_gate_targets(
    direct,
    mapped,
    role_labels,
    target_temperature: float,
    gap_temperature: float,
    torch,
    hard_target: bool = False,
    unweighted: bool = False,
):
    """Build detached oracle routing targets from the two experts' true-class losses."""
    if direct.shape != mapped.shape:
        raise ValueError("Direct and mapped probabilities must have the same shape.")
    _validate_probability_matrix(direct, "direct")
    if role_labels.ndim != 1 or role_labels.shape[0] != direct.shape[0]:
        raise ValueError("Role labels must have shape [batch].")
    if target_temperature <= 0 or gap_temperature <= 0:
        raise ValueError("Gate temperatures must be positive.")
    if bool(((role_labels < 0) | (role_labels >= direct.shape[1])).any()):
        raise ValueError("Role labels contain an index outside the role dimension.")

    epsilon = torch.finfo(direct.dtype).eps
    gather_index = role_labels.view(-1, 1)
    direct_true = direct.gather(1, gather_index).clamp_min(epsilon)
    mapped_true = mapped.gather(1, gather_index).clamp_min(epsilon)
    expert_loss_gap = mapped_true.log() * -1.0 - direct_true.log() * -1.0
    hard_oracle_gate = (expert_loss_gap >= 0).to(direct.dtype)
    soft_oracle_gate = torch.sigmoid(expert_loss_gap / target_temperature)
    if hard_target:
        soft_oracle_gate = hard_oracle_gate
    gate_gap_weight = torch.ones_like(expert_loss_gap)
    if not unweighted:
        gate_gap_weight = torch.tanh(expert_loss_gap.abs() / gap_temperature)

    return {
        "direct_true_probability": direct_true.detach(),
        "mapped_true_probability": mapped_true.detach(),
        "expert_loss_gap": expert_loss_gap.detach(),
        "soft_oracle_gate": soft_oracle_gate.detach(),
        "hard_oracle_gate": hard_oracle_gate.detach(),
        "gate_gap_weight": gate_gap_weight.detach(),
    }


def weighted_soft_gate_loss(gate, target, weight, torch):
    """Return gap-weighted binary cross entropy for a soft oracle gate target."""
    if gate.ndim != 2 or gate.shape[1] != 1:
        raise ValueError("Gate must have shape [batch, 1].")
    if target.shape != gate.shape or weight.shape != gate.shape:
        raise ValueError("Gate target and weight must have the same shape as gate.")
    if bool(((gate < 0) | (gate > 1)).any()) or bool(((target < 0) | (target > 1)).any()):
        raise ValueError("Gate and target values must lie in [0, 1].")
    if bool((weight < 0).any()):
        raise ValueError("Gate weights must be non-negative.")

    epsilon = torch.finfo(gate.dtype).eps
    weight_sum = weight.sum()
    if float(weight_sum.detach()) <= epsilon:
        return gate.sum() * 0.0
    safe_gate = gate.clamp(min=epsilon, max=1.0 - epsilon)
    row_loss = -(target * safe_gate.log() + (1.0 - target) * (1.0 - safe_gate).log())
    return (weight * row_loss).sum() / weight_sum.clamp_min(epsilon)


def gate_routing_diagnostics(gate, direct, mapped, role_labels, torch):
    """Compute label-dependent routing diagnostics without altering inference outputs."""
    if direct.shape != mapped.shape:
        raise ValueError("Direct and mapped probabilities must have the same shape.")
    _validate_probability_matrix(direct, "direct")
    if gate.shape != (direct.shape[0], 1):
        raise ValueError("Gate must have shape [batch, 1].")
    if role_labels.ndim != 1 or role_labels.shape[0] != direct.shape[0]:
        raise ValueError("Role labels must have shape [batch].")

    epsilon = torch.finfo(direct.dtype).eps
    gather_index = role_labels.view(-1, 1)
    direct_true = direct.gather(1, gather_index).clamp_min(epsilon)
    mapped_true = mapped.gather(1, gather_index).clamp_min(epsilon)
    fused_true = (gate * direct_true + (1.0 - gate) * mapped_true).clamp_min(epsilon)
    oracle_true = torch.maximum(direct_true, mapped_true)
    hard_oracle_gate = (direct_true >= mapped_true).to(direct.dtype)
    hard_gate = (gate >= 0.5).to(direct.dtype)
    direct_correct = direct.argmax(dim=1, keepdim=True) == gather_index
    mapped_correct = mapped.argmax(dim=1, keepdim=True) == gather_index
    one_right_one_wrong = direct_correct != mapped_correct
    absolute_hard_gate_error = (gate - hard_oracle_gate).abs()
    true_probability_gap = (direct_true - mapped_true).abs()

    return {
        "direct_true_probability": direct_true,
        "mapped_true_probability": mapped_true,
        "fused_true_probability": fused_true,
        "hard_oracle_gate": hard_oracle_gate,
        "hard_gate_selection_correct": hard_gate == hard_oracle_gate,
        "direct_correct": direct_correct,
        "mapped_correct": mapped_correct,
        "one_right_one_wrong": one_right_one_wrong,
        "absolute_hard_gate_error": absolute_hard_gate_error,
        "true_probability_gap": true_probability_gap,
        "weighted_gate_error": absolute_hard_gate_error * true_probability_gap,
        "routing_regret_nll": -fused_true.log() + oracle_true.log(),
    }


def _validate_verifier_inputs(
    fused,
    ti_target_probability,
    metallic_target_probability,
) -> None:
    _validate_probability_matrix(fused, "fused")
    expected_shape = (fused.shape[0], 1)
    if ti_target_probability.shape != expected_shape or metallic_target_probability.shape != expected_shape:
        raise ValueError("Verifier probabilities must have shape [batch, 1].")
    for values in (ti_target_probability, metallic_target_probability):
        if bool(((values < 0) | (values > 1)).any()):
            raise ValueError("Verifier probabilities must lie in [0, 1].")


def _apply_target_scale(fused, target_scale, target_index: int):
    if not 0 <= target_index < fused.shape[1]:
        raise ValueError("target_index is outside the role dimension.")
    verifier_scale = fused.new_ones(fused.shape)
    verifier_scale[:, target_index : target_index + 1] = target_scale
    evidence = fused * verifier_scale
    epsilon = evidence.new_tensor(torch.finfo(evidence.dtype).eps)
    return evidence / evidence.sum(dim=1, keepdim=True).clamp_min(epsilon)


def apply_multiplicative_target_verifiers(
    fused,
    ti_target_probability,
    metallic_target_probability,
    target_index: int = 0,
):
    """Reproduce the registered pilot that directly multiplies target probabilities."""
    _validate_verifier_inputs(fused, ti_target_probability, metallic_target_probability)
    target_scale = ti_target_probability * metallic_target_probability
    return _apply_target_scale(fused, target_scale, target_index)


def apply_residual_target_verifiers(
    fused,
    ti_target_probability,
    metallic_target_probability,
    target_index: int = 0,
    ti_threshold: float = 0.5,
    metallic_threshold: float = 0.5,
    ti_strength: float = 1.0,
    metallic_strength: float = 1.0,
):
    """Apply bounded penalties only when a verifier contradicts the target."""
    _validate_verifier_inputs(fused, ti_target_probability, metallic_target_probability)
    if not 0.0 < ti_threshold <= 1.0 or not 0.0 < metallic_threshold <= 1.0:
        raise ValueError("Verifier thresholds must lie in (0, 1].")
    if ti_strength < 0 or metallic_strength < 0:
        raise ValueError("Verifier strengths must be non-negative.")
    ti_contradiction = functional.relu((ti_threshold - ti_target_probability) / ti_threshold)
    metallic_contradiction = functional.relu(
        (metallic_threshold - metallic_target_probability) / metallic_threshold
    )
    target_scale = torch.exp(
        -ti_strength * ti_contradiction - metallic_strength * metallic_contradiction
    )
    return _apply_target_scale(fused, target_scale, target_index)


# Backward-compatible name for reproducing the first formal pilot.
apply_target_verifiers = apply_multiplicative_target_verifiers


def masked_verifier_loss(
    logits,
    role_labels,
    negative_role_id: int,
    criterion,
    target_role_id: int = 0,
):
    """Train a target-versus-one-hard-negative verifier on eligible rows only."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Verifier logits must have shape [batch, 2].")
    if role_labels.ndim != 1 or role_labels.shape[0] != logits.shape[0]:
        raise ValueError("Role labels must have shape [batch].")
    eligible = (role_labels == target_role_id) | (role_labels == negative_role_id)
    eligible_count = int(eligible.sum().item())
    if eligible_count == 0:
        return logits.sum() * 0.0, 0
    binary_targets = (role_labels[eligible] == target_role_id).long()
    return criterion(logits[eligible], binary_targets), eligible_count


@dataclass(frozen=True)
class HRGVLossWeights:
    direct: float = 0.25
    species: float = 0.50
    consistency: float = 0.10
    verifier: float = 0.50
    contrast: float = 0.10
    gate_regret: float = 0.0
    decomposition: float = 0.0
    calibration: float = 0.0

    def validate(self) -> None:
        values = (
            self.direct,
            self.species,
            self.consistency,
            self.verifier,
            self.contrast,
            self.gate_regret,
            self.decomposition,
            self.calibration,
        )
        if any(value < 0 for value in values):
            raise ValueError("HRGV loss weights must be non-negative.")


class HierarchicalRiskGatedVerificationNet(nn.Module):
    """EfficientNet-B0 with hierarchical expert fusion and asymmetric verification."""

    def __init__(
        self,
        models,
        role_matrix: torch.Tensor,
        pretrained: bool,
        embedding_dim: int = 128,
        gate_hidden_dim: int = 128,
        fixed_gate: float | None = None,
        disable_verifiers: bool = False,
        verifier_mode: str = "residual",
        ti_verifier_threshold: float = 0.5,
        metallic_verifier_threshold: float = 0.5,
        ti_verifier_strength: float = 1.0,
        metallic_verifier_strength: float = 1.0,
        detach_verifier_features: bool = True,
        detach_gate_features: bool = False,
        enable_cgdc: bool = False,
        cgdc_shared_features: bool = False,
        cgdc_unconditional: bool = False,
        adapter_bottleneck_dim: int = 128,
        calibration_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if verifier_mode not in {"residual", "multiplicative", "disabled"}:
            raise ValueError(
                "verifier_mode must be residual, multiplicative, or disabled."
            )
        if role_matrix.ndim != 2 or role_matrix.shape[0] < 2 or role_matrix.shape[1] < 2:
            raise ValueError("role_matrix must have shape [roles, species].")
        if not torch.allclose(
            role_matrix.sum(dim=0),
            torch.ones(role_matrix.shape[1], dtype=role_matrix.dtype, device=role_matrix.device),
        ):
            raise ValueError("Every species must map to exactly one role.")
        if bool((role_matrix < 0).any()):
            raise ValueError("role_matrix must be non-negative.")
        if min(embedding_dim, gate_hidden_dim, adapter_bottleneck_dim, calibration_hidden_dim) <= 0:
            raise ValueError("Network hidden dimensions must be positive.")
        if fixed_gate is not None and not 0.0 <= fixed_gate <= 1.0:
            raise ValueError("fixed_gate must lie in [0, 1].")
        if not 0.0 < ti_verifier_threshold <= 1.0 or not 0.0 < metallic_verifier_threshold <= 1.0:
            raise ValueError("Verifier thresholds must lie in (0, 1].")
        if ti_verifier_strength < 0 or metallic_verifier_strength < 0:
            raise ValueError("Verifier strengths must be non-negative.")

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.dropout = nn.Dropout(p=base_model.classifier[0].p, inplace=False)
        feature_dim = base_model.classifier[1].in_features
        num_roles, num_species = role_matrix.shape
        self.role_head = nn.Linear(feature_dim, num_roles)
        self.species_head = nn.Linear(feature_dim, num_species)
        self.enable_cgdc = enable_cgdc
        self.cgdc_shared_features = cgdc_shared_features
        self.cgdc_unconditional = cgdc_unconditional
        if enable_cgdc and not cgdc_shared_features:
            self.direct_adapter = nn.Sequential(
                nn.Linear(feature_dim, adapter_bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Linear(adapter_bottleneck_dim, feature_dim),
            )
            self.species_adapter = nn.Sequential(
                nn.Linear(feature_dim, adapter_bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Linear(adapter_bottleneck_dim, feature_dim),
            )
        else:
            self.direct_adapter = None
            self.species_adapter = None
        calibration_input_dim = feature_dim * 4 + num_roles + 3
        self.calibration_network = (
            nn.Sequential(
                nn.Linear(calibration_input_dim, calibration_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(calibration_hidden_dim, num_roles),
            )
            if enable_cgdc
            else None
        )
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim + 3, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.ti_verifier_head = nn.Linear(feature_dim, 2)
        self.metallic_verifier_head = nn.Linear(feature_dim, 2)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, embedding_dim),
        )
        self.fixed_gate = fixed_gate
        self.verifier_mode = "disabled" if disable_verifiers else verifier_mode
        self.disable_verifiers = self.verifier_mode == "disabled"
        self.ti_verifier_threshold = ti_verifier_threshold
        self.metallic_verifier_threshold = metallic_verifier_threshold
        self.ti_verifier_strength = ti_verifier_strength
        self.metallic_verifier_strength = metallic_verifier_strength
        self.detach_verifier_features = detach_verifier_features
        self.detach_gate_features = detach_gate_features
        self.register_buffer("role_matrix", role_matrix.detach().clone().to(dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = torch.flatten(self.avgpool(self.features(images)), 1)
        dropped_features = self.dropout(features)
        if self.enable_cgdc:
            if self.direct_adapter is None or self.species_adapter is None:
                direct_adapter_delta = torch.zeros_like(dropped_features)
                species_adapter_delta = torch.zeros_like(dropped_features)
            else:
                direct_adapter_delta = self.direct_adapter(dropped_features)
                species_adapter_delta = self.species_adapter(dropped_features)
            direct_features = dropped_features + direct_adapter_delta
            species_features = dropped_features + species_adapter_delta
        else:
            direct_features = dropped_features
            species_features = dropped_features
        role_logits = self.role_head(direct_features)
        species_logits = self.species_head(species_features)
        direct_role_probabilities = torch.softmax(role_logits, dim=1)
        species_probabilities = torch.softmax(species_logits, dim=1)
        mapped_role_probabilities = species_probabilities @ self.role_matrix.T
        mapped_role_probabilities = mapped_role_probabilities / mapped_role_probabilities.sum(
            dim=1, keepdim=True
        ).clamp_min(torch.finfo(mapped_role_probabilities.dtype).eps)
        direct_entropy = normalized_entropy(direct_role_probabilities, torch)
        mapped_entropy = normalized_entropy(mapped_role_probabilities, torch)
        expert_js_divergence = jensen_shannon_divergence(
            direct_role_probabilities, mapped_role_probabilities, torch
        )
        if self.fixed_gate is None:
            gate_inputs = torch.cat(
                [features, direct_entropy, mapped_entropy, expert_js_divergence], dim=1
            )
            if self.detach_gate_features:
                gate_inputs = gate_inputs.detach()
            gate = torch.sigmoid(self.gate_network(gate_inputs))
        else:
            gate = features.new_full((features.shape[0], 1), self.fixed_gate)
        fused_role_probabilities = mix_role_experts(
            direct_role_probabilities, mapped_role_probabilities, gate
        )
        if self.enable_cgdc:
            if self.calibration_network is None:
                raise RuntimeError("CGDC calibration network is not initialized.")
            calibration_inputs = torch.cat(
                [
                    direct_features,
                    species_features,
                    direct_features * species_features,
                    (direct_features - species_features).abs(),
                    direct_role_probabilities.clamp_min(torch.finfo(direct_role_probabilities.dtype).eps).log()
                    - mapped_role_probabilities.clamp_min(torch.finfo(mapped_role_probabilities.dtype).eps).log(),
                    direct_entropy,
                    mapped_entropy,
                    expert_js_divergence,
                ],
                dim=1,
            )
            calibration_residual = self.calibration_network(calibration_inputs)
            if self.cgdc_unconditional:
                disagreement_gain = torch.ones_like(expert_js_divergence)
                calibrated_logits = (
                    fused_role_probabilities.clamp_min(
                        torch.finfo(fused_role_probabilities.dtype).eps
                    ).log()
                    + torch.tanh(calibration_residual)
                )
                calibrated_role_probabilities = torch.softmax(calibrated_logits, dim=1)
            else:
                calibrated_role_probabilities, disagreement_gain = (
                    disagreement_calibrated_role_probabilities(
                        fused_role_probabilities,
                        direct_role_probabilities,
                        mapped_role_probabilities,
                        calibration_residual,
                        torch,
                    )
                )
        else:
            calibrated_role_probabilities = fused_role_probabilities
        verifier_features = features.detach() if self.detach_verifier_features else features
        ti_verifier_logits = self.ti_verifier_head(verifier_features)
        metallic_verifier_logits = self.metallic_verifier_head(verifier_features)
        ti_target_probability = torch.softmax(ti_verifier_logits, dim=1)[:, 1:2]
        metallic_target_probability = torch.softmax(metallic_verifier_logits, dim=1)[:, 1:2]
        if self.verifier_mode == "disabled":
            final_role_probabilities = calibrated_role_probabilities
        elif self.verifier_mode == "multiplicative":
            final_role_probabilities = apply_multiplicative_target_verifiers(
                calibrated_role_probabilities,
                ti_target_probability,
                metallic_target_probability,
            )
        else:
            final_role_probabilities = apply_residual_target_verifiers(
                calibrated_role_probabilities,
                ti_target_probability,
                metallic_target_probability,
                ti_threshold=self.ti_verifier_threshold,
                metallic_threshold=self.metallic_verifier_threshold,
                ti_strength=self.ti_verifier_strength,
                metallic_strength=self.metallic_verifier_strength,
            )
        embeddings = functional.normalize(self.projection_head(features), p=2, dim=1)
        outputs = {
            "role_logits": role_logits,
            "species_logits": species_logits,
            "direct_role_probabilities": direct_role_probabilities,
            "mapped_role_probabilities": mapped_role_probabilities,
            "gate": gate,
            "fused_role_probabilities": fused_role_probabilities,
            "ti_verifier_logits": ti_verifier_logits,
            "metallic_verifier_logits": metallic_verifier_logits,
            "ti_target_probability": ti_target_probability,
            "metallic_target_probability": metallic_target_probability,
            "final_role_probabilities": final_role_probabilities,
            "embeddings": embeddings,
            "expert_js_divergence": expert_js_divergence,
        }
        if self.enable_cgdc:
            outputs.update(
                {
                    "direct_adapter_delta": direct_adapter_delta,
                    "species_adapter_delta": species_adapter_delta,
                    "calibration_residual": torch.tanh(calibration_residual),
                    "disagreement_gain": disagreement_gain,
                    "calibrated_role_probabilities": calibrated_role_probabilities,
                }
            )
        return outputs


def compute_hrgv_losses(
    outputs,
    role_labels,
    species_labels,
    mapping,
    final_role_criterion,
    direct_role_criterion,
    species_criterion,
    verifier_criterion,
    weights: HRGVLossWeights,
    temperature: float,
    torch,
    gate_target_temperature: float = 0.20,
    gate_gap_temperature: float = 0.50,
    hard_gate_target: bool = False,
    unweighted_gate_regret: bool = False,
):
    """Return the complete HRGV objective and its auditable loss terms."""
    weights.validate()
    if outputs["species_logits"].shape[1] != len(mapping.species_labels):
        raise ValueError("Species logits do not match the frozen mapping.")
    epsilon = torch.finfo(outputs["final_role_probabilities"].dtype).eps
    final_role_loss = final_role_criterion(
        outputs["final_role_probabilities"].clamp_min(epsilon).log(), role_labels
    )
    direct_role_loss = direct_role_criterion(outputs["role_logits"], role_labels)
    species_loss = species_criterion(outputs["species_logits"], species_labels)
    consistency_loss = functional.kl_div(
        outputs["direct_role_probabilities"].clamp_min(epsilon).log(),
        outputs["mapped_role_probabilities"],
        reduction="batchmean",
    )
    ti_verifier_loss, _ = masked_verifier_loss(
        outputs["ti_verifier_logits"],
        role_labels,
        negative_role_id=1,
        criterion=verifier_criterion,
    )
    metallic_verifier_loss, _ = masked_verifier_loss(
        outputs["metallic_verifier_logits"],
        role_labels,
        negative_role_id=3,
        criterion=verifier_criterion,
    )
    verifier_loss = ti_verifier_loss + metallic_verifier_loss
    contrast_loss = compute_role_aware_contrastive_loss(
        outputs["embeddings"], role_labels, temperature, torch
    )
    gate_targets = regret_gate_targets(
        outputs["direct_role_probabilities"],
        outputs["mapped_role_probabilities"],
        role_labels,
        target_temperature=gate_target_temperature,
        gap_temperature=gate_gap_temperature,
        torch=torch,
        hard_target=hard_gate_target,
        unweighted=unweighted_gate_regret,
    )
    gate_regret_loss = weighted_soft_gate_loss(
        outputs["gate"],
        gate_targets["soft_oracle_gate"],
        gate_targets["gate_gap_weight"],
        torch,
    )
    calibrated_role_probabilities = outputs.get(
        "calibrated_role_probabilities", outputs["fused_role_probabilities"]
    )
    calibration_loss = final_role_criterion(
        calibrated_role_probabilities.clamp_min(epsilon).log(), role_labels
    )
    direct_delta = outputs.get("direct_adapter_delta")
    species_delta = outputs.get("species_adapter_delta")
    if direct_delta is None or species_delta is None:
        decomposition_loss = outputs["final_role_probabilities"].sum() * 0.0
    else:
        decomposition_loss = adapter_decomposition_loss(direct_delta, species_delta, torch)
    total_loss = (
        final_role_loss
        + weights.direct * direct_role_loss
        + weights.species * species_loss
        + weights.consistency * consistency_loss
        + weights.verifier * verifier_loss
        + weights.contrast * contrast_loss
        + weights.gate_regret * gate_regret_loss
        + weights.decomposition * decomposition_loss
        + weights.calibration * calibration_loss
    )
    return total_loss, {
        "final_role_loss": final_role_loss,
        "direct_role_loss": direct_role_loss,
        "species_loss": species_loss,
        "consistency_loss": consistency_loss,
        "ti_verifier_loss": ti_verifier_loss,
        "metallic_verifier_loss": metallic_verifier_loss,
        "verifier_loss": verifier_loss,
        "contrast_loss": contrast_loss,
        "gate_regret_loss": gate_regret_loss,
        "decomposition_loss": decomposition_loss,
        "calibration_loss": calibration_loss,
    }
