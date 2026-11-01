"""TopoSAS: skeleton-free STCL measurement and fixed-budget patch planning."""

from __future__ import annotations

__version__ = "0.1.0"

from .patch import (  # noqa: E402
    PatchAllocation,
    fit_patch,
    fit_patches,
    pairwise_log_ratio_error,
    plan_permutation,
    reorder_plan_to_xyz,
    reorder_xyz_to_plan,
)
from .presets import (  # noqa: E402
    PRESETS,
    DatasetPreset,
    reproduce_all,
    reproduce_preset,
)
from .stcl import (  # noqa: E402
    DEFAULT_COVERAGE,
    STCLMeasurement,
    aggregate_measurements,
    measure_label,
    measure_nifti,
    stcl_xyz_mm,
)

__all__ = [
    "DEFAULT_COVERAGE",
    "PRESETS",
    "DatasetPreset",
    "PatchAllocation",
    "STCLMeasurement",
    "__version__",
    "aggregate_measurements",
    "fit_patch",
    "fit_patches",
    "measure_label",
    "measure_nifti",
    "pairwise_log_ratio_error",
    "plan_permutation",
    "reorder_plan_to_xyz",
    "reorder_xyz_to_plan",
    "reproduce_all",
    "reproduce_preset",
    "stcl_xyz_mm",
]
