from __future__ import annotations

from dataclasses import dataclass

from train_mineral_classifier import CLASS_TO_ID


@dataclass(frozen=True)
class SpeciesRoleMapping:
    """Stable mineral-species indices and their four-role assignments."""

    species_labels: tuple[str, ...]
    species_role_ids: tuple[int, ...]
    species_to_index: dict[str, int]


def build_species_mapping(records) -> SpeciesRoleMapping:
    """Build a stable species-to-role mapping and reject ambiguous manifest labels."""
    species_to_role: dict[str, str] = {}
    for record in records:
        species = record.mineral_label.strip()
        role = record.four_class_label.strip()
        if not species:
            raise ValueError("Mineral labels must not be empty.")
        if role not in CLASS_TO_ID:
            raise ValueError(f"Unknown four-class role for {species}: {role}")
        prior_role = species_to_role.setdefault(species, role)
        if prior_role != role:
            raise ValueError(f"Species {species} is assigned to multiple roles: {prior_role}, {role}")

    species_labels = tuple(sorted(species_to_role))
    if not species_labels:
        raise ValueError("At least one mineral species is required.")
    species_role_ids = tuple(CLASS_TO_ID[species_to_role[species]] for species in species_labels)
    return SpeciesRoleMapping(
        species_labels=species_labels,
        species_role_ids=species_role_ids,
        species_to_index={species: index for index, species in enumerate(species_labels)},
    )


def validate_species_role_mapping(records) -> SpeciesRoleMapping:
    """Validate and return the frozen manifest's species-to-role mapping."""
    return build_species_mapping(records)


def aggregate_role_probabilities(species_probabilities, mapping: SpeciesRoleMapping, torch):
    """Aggregate species posterior probabilities into the fixed four-role posterior."""
    if species_probabilities.ndim != 2:
        raise ValueError("Species probabilities must be a two-dimensional tensor.")
    if species_probabilities.size(1) != len(mapping.species_labels):
        raise ValueError("Species probability width does not match the mapping.")
    aggregation = torch.zeros(
        (len(CLASS_TO_ID), len(mapping.species_labels)),
        dtype=species_probabilities.dtype,
        device=species_probabilities.device,
    )
    aggregation[mapping.species_role_ids, torch.arange(len(mapping.species_labels), device=species_probabilities.device)] = 1
    return species_probabilities @ aggregation.T
