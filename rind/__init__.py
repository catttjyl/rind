"""Unified Clean-Room Full-Day RIND Package (MANHATTAN_4445_ANNUAL_18808_V1).

Single authoritative implementation for:
- Strict 1km-block-disjoint spatial split, tri-state exposure firewall & blind-test lock (`split_contract`)
- Real host physical RAM / commit / CUDA VRAM telemetry & fail-closed resource guard (`system_guard`)
- True 52-week sunrise-to-sunset (`18,808` frames/tile) full-day streaming reader & 4km geometry builder (`fullday_data`)
- Provenance-bound contiguous mmap shards & day/receiver streaming pipeline (`fast_data`)
- Unified self-contained neural model zoo with isolated geometry tiers & bounded recurrence (`models`)
- Streaming chunked `bf16`/`fp32` hybrid full-day training, calibration & 3-arm speedup benchmark (`fast_train`)
- Exact matched-denominator state, transition, distance-band & paired bootstrap metrics (`metrics`)
- Full-scene 4km exact double-precision BVH oracle & volume-conserving full-day inverse design (`inverse_design`)
"""

from __future__ import annotations

BENCHMARK_CONTRACT_ID = "MANHATTAN_4445_ANNUAL_18808_V1"
FEATURE_SCHEMA_VERSION = "canonical37_v2"
PACKAGE_VERSION = "2.2.0rc5"

__all__ = [
    "BENCHMARK_CONTRACT_ID",
    "FEATURE_SCHEMA_VERSION",
    "PACKAGE_VERSION",
    "ObjectiveConfig",
    "evaluate_representation_sufficiency",
    "infer_unknown_building_heights_bayesian",
]


def __getattr__(name: str):
    if name == "ObjectiveConfig":
        from .objectives_jepa import ObjectiveConfig

        return ObjectiveConfig
    if name == "evaluate_representation_sufficiency":
        from .inverse_design import evaluate_representation_sufficiency

        return evaluate_representation_sufficiency
    if name == "infer_unknown_building_heights_bayesian":
        from .inverse_design import infer_unknown_building_heights_bayesian

        return infer_unknown_building_heights_bayesian
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
