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

    def validate(self) -> None:
        values = (self.direct, self.species, self.consistency, self.verifier, self.contrast)
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
        if embedding_dim <= 0 or gate_hidden_dim <= 0:
            raise ValueError("Embedding and gate hidden dimensions must be positive.")
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
        self.register_buffer("role_matrix", role_matrix.detach().clone().to(dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = torch.flatten(self.avgpool(self.features(images)), 1)
        dropped_features = self.dropout(features)
        role_logits = self.role_head(dropped_features)
        species_logits = self.species_head(dropped_features)
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
            gate = torch.sigmoid(self.gate_network(gate_inputs))
        else:
            gate = features.new_full((features.shape[0], 1), self.fixed_gate)
        fused_role_probabilities = mix_role_experts(
            direct_role_probabilities, mapped_role_probabilities, gate
        )
        verifier_features = features.detach() if self.detach_verifier_features else features
        ti_verifier_logits = self.ti_verifier_head(verifier_features)
        metallic_verifier_logits = self.metallic_verifier_head(verifier_features)
        ti_target_probability = torch.softmax(ti_verifier_logits, dim=1)[:, 1:2]
        metallic_target_probability = torch.softmax(metallic_verifier_logits, dim=1)[:, 1:2]
        if self.verifier_mode == "disabled":
            final_role_probabilities = fused_role_probabilities
        elif self.verifier_mode == "multiplicative":
            final_role_probabilities = apply_multiplicative_target_verifiers(
                fused_role_probabilities,
                ti_target_probability,
                metallic_target_probability,
            )
        else:
            final_role_probabilities = apply_residual_target_verifiers(
                fused_role_probabilities,
                ti_target_probability,
                metallic_target_probability,
                ti_threshold=self.ti_verifier_threshold,
                metallic_threshold=self.metallic_verifier_threshold,
                ti_strength=self.ti_verifier_strength,
                metallic_strength=self.metallic_verifier_strength,
            )
        embeddings = functional.normalize(self.projection_head(features), p=2, dim=1)
        return {
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
    total_loss = (
        final_role_loss
        + weights.direct * direct_role_loss
        + weights.species * species_loss
        + weights.consistency * consistency_loss
        + weights.verifier * verifier_loss
        + weights.contrast * contrast_loss
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
    }
