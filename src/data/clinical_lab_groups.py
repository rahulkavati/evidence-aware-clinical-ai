"""Clinical grouping and group-level masking for all laboratory features."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

from src.data.build_temporal_features import LAB_FEATURES
from src.data.g1_masking import apply_hidden_set


CLINICAL_LAB_GROUPS: Mapping[str, tuple[str, ...]] = {
    "blood_gas": (
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2",
    ),
    "inflammation_perfusion": (
        "Lactate", "WBC", "Fibrinogen",
    ),
    "organ_function": (
        "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
        "Creatinine", "Bilirubin_direct", "Glucose", "Magnesium",
        "Phosphate", "Potassium", "Bilirubin_total", "TroponinI",
    ),
    "hematology_coagulation": (
        "Hct", "Hgb", "PTT", "Platelets",
    ),
}


def validate_lab_groups(
    groups: Mapping[str, Sequence[str]] = CLINICAL_LAB_GROUPS,
) -> None:
    """Require that groups partition the repository's 26 laboratory features."""
    names = [lab for labs in groups.values() for lab in labs]
    if len(names) != len(set(names)):
        raise ValueError("A laboratory feature appears in multiple groups")
    if set(names) != set(LAB_FEATURES):
        missing = sorted(set(LAB_FEATURES) - set(names))
        extra = sorted(set(names) - set(LAB_FEATURES))
        raise ValueError(f"Groups do not partition laboratory features; missing={missing}, extra={extra}")


def labs_for_groups(
    group_names: Iterable[str],
    groups: Mapping[str, Sequence[str]] = CLINICAL_LAB_GROUPS,
) -> tuple[str, ...]:
    """Expand group names to a stable, duplicate-free laboratory tuple."""
    validate_lab_groups(groups)
    selected: list[str] = []
    for group_name in group_names:
        if group_name not in groups:
            raise KeyError(f"Unknown laboratory group: {group_name}")
        selected.extend(groups[group_name])
    if len(selected) != len(set(selected)):
        raise ValueError("A laboratory was selected through multiple group names")
    return tuple(selected)


def apply_hidden_groups(
    frame: pd.DataFrame,
    group_names: Iterable[str],
    groups: Mapping[str, Sequence[str]] = CLINICAL_LAB_GROUPS,
) -> pd.DataFrame:
    """Hide every laboratory in the requested clinical groups."""
    return apply_hidden_set(frame, labs_for_groups(group_names, groups))


validate_lab_groups()
