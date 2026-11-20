"""Unified Command-Line Interface (`python -m rind <subcommand>`).

Provides deterministic, firewall-guarded, streaming subcommands for the canonical
`MANHATTAN_4445_ANNUAL_18808_V1` full-day urban solar-shadow benchmark:
  - `verify-split`
  - `inspect-annual-tile`
  - `build-mmap-cache`
  - `train`
  - `calibrate`
  - `eval` (`eval-fullday` alias; strictly `val_spatial_clean` during development)
  - `eval-legacy` (`test_spatial_legacy_reassigned`, gated on scratch `fit`-only lineage)
  - `unseal-blind-test` (`test_spatial_unexposed`, gated on frozen pre-registration manifest)
  - `benchmark-latency` (matched-domain neural vs exact `4 km` `ExactSceneBVH` ray-tracing)
  - `benchmark-train-speed` (3-arm FP32/BF16 matched speedup + shard amortization)
  - `inverse-design` (full `4 km` BVH physical oracle + 3 optimizer arms)
  - `run-dev-suite` (`run-master-suite` alias; keeps `blind_open_count == 0`)
"""

from __future__ import annotations

import argparse
import builtins
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from . import BENCHMARK_CONTRACT_ID, FEATURE_SCHEMA_VERSION, PACKAGE_VERSION
from .fast_data import (
    ContiguousTileMmapShard,
    _read_npy_slice_no_mmap,
    _write_npy_header_bytes,
    build_or_open_mmap_cache,
    build_or_open_single_tile_shard,
    get_fast_data_io_stats,
    materialize_gpu_tokens_from_azimuth_bank,
    reset_fast_data_io_stats,
)
from .fast_train import (
    LabelReadForbiddenBatchView,
    benchmark_training_speed,
    calibrate_model_on_split,
    compute_long_day_lag_diagnostics,
    directed_event_matches,
    fit_model_fullday,
    get_process_rss_mib,
    move_batch_to_device,
)
from .fullday_data import (
    DEFAULT_SCENE_HORIZON_M,
    FullDayTileReader,
    resolve_week_indices,
)
from .metrics import (
    DISTANCE_BAND_ORDER,
    SOLAR_REGIME_ORDER,
    ConfusionAccumulator,
    aggregate_multiseed_reports,
    bootstrap_tile_ci_95,
    compute_blocker_distance_stratified_metrics,
    compute_shadow_contour_f1,
    evaluate_trajectory_predictions,
    paired_bootstrap_model_comparison,
)
from .models import (
    ALL_MODEL_TYPES,
    ATTENTION_MODES,
    INFERENCE_MODES,
    MODEL_TYPES,
    build_model,
    compute_head_parameter_norms,
    load_rind_checkpoint,
    save_rind_checkpoint,
)
from .objectives_jepa import (
    ALL_MASK_MODES,
    EMA_TARGET_SCOPES,
    JEPA_CONTEXT_MODES,
    JEPA_MODES,
    REG_MODES,
    TRANSITION_MODES,
    ObjectiveConfig,
)
from .split_contract import (
    CANONICAL_ALL_BUCKETS,
    CANONICAL_EVAL_BUCKETS,
    FIT_SUBSET_SIZES,
    SplitContract,
    atomic_update_json_file,
    compute_file_sha256,
)
from .system_guard import HostResourceGuard


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path | str, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False, allow_nan=False), encoding="utf-8")
    tmp.replace(out)
    return out


def _append_val_exposure_ledger(
    ledger_path: Path,
    entry: Mapping[str, Any],
) -> Dict[str, Any]:
    default_doc = {
        "schema": "rind.val_exposure_ledger.v1",
        "benchmark_contract": BENCHMARK_CONTRACT_ID,
        "entries": [],
    }

    def _mutator(raw_doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        doc = dict(raw_doc) if isinstance(raw_doc, dict) else dict(default_doc)
        entries = list(doc.get("entries", []))
        rec = {"eval_sequence_index": len(entries) + 1, **dict(entry)}
        entries.append(rec)
        doc["schema"] = "rind.val_exposure_ledger.v1"
        doc["benchmark_contract"] = BENCHMARK_CONTRACT_ID
        doc["entries"] = entries
        doc["total_val_evaluations"] = len(entries)
        doc["updated_at_utc"] = _utc_now_iso()
        return doc

    return atomic_update_json_file(ledger_path, _mutator, default_factory=lambda: dict(default_doc))


def _add_common_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split-file", type=Path, default=None, help="Path to MANHATTAN_4445_SPLIT_V1.json")
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=None,
        help="Path to ANNUAL_FULLDAY_18808_MANIFEST_V1.json",
    )
    parser.add_argument(
        "--allow-synthetic-split",
        action="store_true",
        help="Allow non-pinned split SHA-256 ONLY when paired with --diagnostic-mode.",
    )
    parser.add_argument(
        "--diagnostic-mode",
        action="store_true",
        help="Explicitly allow non-annual protocols, --max-tiles > 0, or synthetic fixtures (stamps DIAGNOSTIC_NON_PUBLICATION).",
    )
    parser.add_argument(
        "--global-mesh-dir",
        type=Path,
        default=None,
        help="Optional explicit path to shared 4km DS3MESH directory.",
    )


def _load_contract_from_args(args: argparse.Namespace) -> SplitContract:
    return SplitContract(
        split_path=getattr(args, "split_file", None),
        manifest_path=getattr(args, "manifest_file", None),
        enforce_pinned_split_sha256=not bool(getattr(args, "allow_synthetic_split", False)),
    )


def _configure_portable_reader(reader, args):
    index_path = getattr(args, "portable_index", None)
    digest = getattr(args, "portable_index_sha256", None)
    if bool(index_path) != bool(digest):
        raise ValueError("--portable-index and --portable-index-sha256 must be provided together")
    if index_path:
        digest = str(digest).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid portable index SHA-256")
        index_bytes = Path(index_path).read_bytes()
        if hashlib.sha256(index_bytes).hexdigest() != digest:
            raise RuntimeError("Portable index SHA-256 mismatch")
        index = json.loads(index_bytes)
        if index.get("schema") != "rind.portable_index.v1" or not isinstance(index.get("tiles"), dict):
            raise RuntimeError("Invalid portable index schema")
        if index.get("split_sha256") != reader.contract.split_sha256 or index.get("manifest_sha256") != reader.contract.manifest_sha256:
            raise RuntimeError("Portable index split/manifest identity mismatch")
        reader.portable_index = index
    return reader


def _add_portable_args(parser):
    parser.add_argument("--portable-index", type=Path, default=None, help="Read-only indexed shards; never rebuild")
    parser.add_argument("--portable-index-sha256", default=None, help="Independently pinned SHA-256 of index")


def _enforce_publication_or_diagnostic_mode(
    args: argparse.Namespace,
    contract: SplitContract,
) -> str:
    """Refuse to stamp a formal publication receipt when non-annual, sampled, truncated, or synthetic flags are active."""
    diag_reasons: List[str] = []
    tp = str(getattr(args, "temporal_protocol", "annual_52week"))
    if tp != "annual_52week":
        diag_reasons.append(f"temporal_protocol={tp!r}!=annual_52week")
    rm = str(getattr(args, "receiver_mode", "all_valid"))
    if rm != "all_valid":
        diag_reasons.append(f"receiver_mode={rm!r}!=all_valid")
    if bool(getattr(args, "allow_synthetic_split", False)):
        diag_reasons.append("allow_synthetic_split=True")
    if int(getattr(args, "max_tiles", 0) or 0) > 0:
        diag_reasons.append(f"max_tiles={int(getattr(args, 'max_tiles', 0))}>0")
    if contract.is_synthetic_fixture:
        diag_reasons.append(f"fixture_kind={contract.fixture_kind!r}")

    diag_flag = bool(getattr(args, "diagnostic_mode", False))
    if diag_reasons and not diag_flag:
        raise RuntimeError(
            "FORMAL PUBLICATION GUARD BLOCKED EXECUTION: non-annual, sampled-receiver, or truncated configuration "
            f"({', '.join(diag_reasons)}) is forbidden in formal mode! Pass `--receiver-mode all_valid` for "
            "formal annual publication receipts, or pass `--diagnostic-mode` explicitly for diagnostic runs."
        )
    if diag_reasons or diag_flag:
        return "DIAGNOSTIC_NON_PUBLICATION"
    return "FORMAL_ANNUAL_PUBLICATION"


def cmd_verify_split(args: argparse.Namespace) -> Dict[str, Any]:
    contract = _load_contract_from_args(args)
    report = contract.build_verification_report()
    report["verified_at_utc"] = _utc_now_iso()
    if args.output:
        _write_json_atomic(args.output, report)
    if not args.quiet:
        print(json.dumps(report, indent=2))
    return report


def cmd_inspect_annual_tile(args: argparse.Namespace) -> Dict[str, Any]:
    contract = _load_contract_from_args(args)
    reader = FullDayTileReader(contract, global_mesh_shared_dir=args.global_mesh_dir)
    target_tile = args.tile_id if args.tile_id else contract.get_fit_tiles("fit_8")[0]
    if target_tile in contract.tiles and contract.tiles[target_tile].canonical_split == "test_spatial_unexposed":
        raise RuntimeError(f"BLIND TEST LOCK: cannot inspect blind tile {target_tile!r} during development!")

    decode_sample = [0, 13, 25, 38] if args.decode_sample_weeks else None
    summary = reader.inspect_tile_annual_summary(target_tile, decode_week_sample=decode_sample)
    summary["benchmark_contract"] = BENCHMARK_CONTRACT_ID
    summary["inspected_at_utc"] = _utc_now_iso()
    if args.output:
        _write_json_atomic(args.output, summary)
    if not args.quiet:
        print(json.dumps(summary, indent=2))
    return summary


def cmd_build_mmap_cache(args: argparse.Namespace) -> Dict[str, Any]:
    contract = _load_contract_from_args(args)
    receipt_class = _enforce_publication_or_diagnostic_mode(args, contract)
    guard = HostResourceGuard()
    guard.check("build_mmap_cache:init")

    reader = FullDayTileReader(contract, global_mesh_shared_dir=args.global_mesh_dir)
    week_indices = resolve_week_indices(args.temporal_protocol)

    if args.split in FIT_SUBSET_SIZES:
        tile_ids = contract.get_fit_tiles(args.split)
    elif args.split == "calibration":
        tile_ids = contract.get_calibration_tiles()
    elif args.split == "val_spatial_clean" or args.split == "test_spatial_legacy_reassigned":
        tile_ids = contract.get_eval_tiles(args.split, allow_empty=True)
    elif args.split == "test_spatial_unexposed":
        raise RuntimeError(
            "BLIND TEST LOCK: pre-building shards for `test_spatial_unexposed` is blocked during development!"
        )
    else:
        raise ValueError(f"Unsupported split {args.split!r} for mmap cache build.")

    if args.max_tiles > 0:
        tile_ids = tile_ids[: int(args.max_tiles)]

    t0 = time.perf_counter()
    total_bytes = 0
    for tid in tile_ids:
        guard.check(f"build_mmap_cache:{tid}")
        sh = build_or_open_single_tile_shard(
            args.cache_dir,
            reader,
            tid,
            week_indices=week_indices,
            receiver_mode=args.receiver_mode,
            max_receivers=args.max_receivers,
            max_candidates=args.max_candidates,
            num_azimuth_bins=args.num_azimuth_bins,
            bbox_margin_m=args.bbox_margin_m,
        )
        total_bytes += int(sh.provenance["total_shard_bytes"])
    elapsed = time.perf_counter() - t0
    guard.check("build_mmap_cache:done")

    receipt = {
        "benchmark_contract": BENCHMARK_CONTRACT_ID,
        "receipt_class": receipt_class,
        "split": args.split,
        "temporal_protocol": args.temporal_protocol,
        "receiver_mode": args.receiver_mode,
        "tile_count": len(tile_ids),
        "total_shard_bytes": total_bytes,
        "wall_seconds": elapsed,
        "resource_telemetry": guard.summary(),
        "created_at_utc": _utc_now_iso(),
    }
    if args.output:
        _write_json_atomic(args.output, receipt)
    if not args.quiet:
        print(json.dumps(receipt, indent=2))
    return receipt


def _profile_switch(args: argparse.Namespace, name: str, default: bool) -> bool:
    value = getattr(args, name, None)
    return default if value is None else bool(value)


def _build_train_objective_config(args: argparse.Namespace) -> ObjectiveConfig:
    """Resolve and validate the training ObjectiveConfig from CLI arguments."""
    matched_arm = getattr(args, "matched_objective", None)
    raw_lb_frac = getattr(args, "label_budget_fraction", 1.0)
    label_budget_frac = 1.0 if raw_lb_frac is None else float(raw_lb_frac)
    raw_lb_seed = getattr(args, "label_budget_seed", 20261112)
    label_budget_seed = 20261112 if raw_lb_seed is None else int(raw_lb_seed)
    mask_mode = str(getattr(args, "mask_mode", "spatial_boundary") or "spatial_boundary")
    mask_ratio = float(
        getattr(args, "mask_ratio", 0.25) if getattr(args, "mask_ratio", None) is not None else 0.25
    )
    buffer_m = float(getattr(args, "buffer_m", 1.5))
    jepa_context_mode = str(getattr(args, "jepa_context_mode", "with_x") or "with_x")

    if matched_arm:
        if args.model_type != "rind_full_geom":
            raise ValueError("Matched objectives share the rind_full_geom backbone")
        modes = {
            "direct": "none",
            "geometry_mae": "geometry_mae",
            "geometry_jepa": "ema_geom",
            "response_jepa": "response",
        }
        return ObjectiveConfig(
            jepa_mode=modes[matched_arm],
            pure_supervised=matched_arm == "direct",
            reg_mode="none",
            transition_mode="none",
            counterfactual_weight=0.0,
            dual_path_weight=0.0,
            dominant_occluder_dropout_prob=0.0,
            ema_target_scope="full_backbone",
            mask_mode=mask_mode,
            jepa_context_mode=jepa_context_mode,
            buffer_m=buffer_m,
            mask_ratio=mask_ratio,
            label_budget_fraction=label_budget_frac,
            label_budget_seed=label_budget_seed,
        )

    is_rind = str(args.model_type).startswith("rind_")
    if not is_rind:
        return ObjectiveConfig(
            jepa_mode="none",
            reg_mode="none",
            transition_mode="none",
            dual_path_weight=0.0,
            counterfactual_weight=0.0,
            dominant_occluder_dropout_prob=0.0,
            label_budget_fraction=label_budget_frac,
            label_budget_seed=label_budget_seed,
            pure_supervised=True,
        )

    jepa_mode = str(getattr(args, "jepa_mode", "hybrid") or "hybrid")
    reg_mode = str(getattr(args, "reg_mode", "sigreg") or "sigreg")
    pure_sup = bool(getattr(args, "pure_supervised", False))
    transition_mode = str(getattr(args, "transition_mode", "event_aware") or "event_aware")
    counterfactual_weight = float(
        getattr(args, "counterfactual_weight", 0.08)
        if getattr(args, "counterfactual_weight", None) is not None
        else 0.08
    )

    if (
        jepa_mode == "none"
        and reg_mode == "none"
        and not pure_sup
        and (transition_mode != "none" or counterfactual_weight > 0.0)
    ):
        raise ValueError(
            "Ambiguous pure-supervised configuration: `--jepa-mode none --reg-mode none` "
            "without `--pure-supervised` leaves transition/dual-path/counterfactual losses active "
            f"(transition_mode={transition_mode!r}, counterfactual_weight={counterfactual_weight}). "
            "Pass `--pure-supervised` for a clean supervised baseline, or explicitly pass "
            "`--transition-mode none --counterfactual-weight 0`."
        )

    dual_path_weight = (
        0.0
        if (
            pure_sup
            or (
                jepa_mode == "none"
                and reg_mode == "none"
                and transition_mode == "none"
                and counterfactual_weight <= 0.0
            )
        )
        else 0.05
    )

    return ObjectiveConfig(
        jepa_mode=jepa_mode,
        ema_target_scope=str(getattr(args, "ema_target_scope", "head") or "head"),
        mask_mode=mask_mode,
        mask_ratio=mask_ratio,
        buffer_m=buffer_m,
        spatial_block_size=int(getattr(args, "spatial_block_size", 24) or 24),
        jepa_context_mode=jepa_context_mode,
        reg_mode=reg_mode,
        transition_mode=transition_mode,
        dual_path_weight=dual_path_weight,
        counterfactual_weight=counterfactual_weight,
        dominant_occluder_dropout_prob=float(
            getattr(args, "dominant_occluder_dropout_prob", 0.25)
            if getattr(args, "dominant_occluder_dropout_prob", None) is not None
            else 0.25
        ),
        label_budget_fraction=label_budget_frac,
        label_budget_seed=label_budget_seed,
        pure_supervised=pure_sup,
    )


def cmd_train(args: argparse.Namespace) -> Dict[str, Any]:
    mainline = _profile_switch(args, "mainline", args.model_type == "rind_full_geom")
    if mainline and args.model_type != "rind_full_geom":
        raise ValueError("--mainline selects the rind_full_geom training configuration")
    obj_cfg = _build_train_objective_config(args)
    contract = _load_contract_from_args(args)
    receipt_class = _enforce_publication_or_diagnostic_mode(args, contract)
    dev = torch.device(args.device)
    guard = HostResourceGuard(device=dev)
    guard.check("cmd_train:init")

    fit_tiles = contract.get_fit_tiles(args.fit_subset)
    if args.max_tiles > 0:
        fit_tiles = fit_tiles[: int(args.max_tiles)]
    firewall_token = contract.issue_fit_firewall_token(fit_tiles, subset_name=args.fit_subset)

    reader = _configure_portable_reader(FullDayTileReader(contract, global_mesh_shared_dir=args.global_mesh_dir), args)
    week_indices = resolve_week_indices(args.temporal_protocol)

    # Open/verify first tile shard to obtain exact meters_per_pixel without opening all 67 tiles at once
    first_shard = build_or_open_single_tile_shard(
        args.cache_dir,
        reader,
        fit_tiles[0],
        week_indices=week_indices,
        receiver_mode=args.receiver_mode,
        max_receivers=args.max_receivers,
        max_candidates=args.max_candidates,
        num_azimuth_bins=args.num_azimuth_bins,
        bbox_margin_m=args.bbox_margin_m,
    )

    torch.manual_seed(int(args.seed))
    explicit_sp_mode = getattr(args, "spatial_execution_mode", None) or getattr(args, "convgru_mode", None)
    if str(args.model_type) == "full_tile_convgru":
        convgru_mode = "full_tile_512_halo"
    elif explicit_sp_mode:
        convgru_mode = str(explicit_sp_mode)
    else:
        convgru_mode = "per_receiver_crop"

    jepa_mode = str(getattr(args, "jepa_mode", "hybrid") or "hybrid")
    reg_mode = str(getattr(args, "reg_mode", "sigreg") or "sigreg")
    reanchor_every = int(getattr(args, "reanchor_every", 0) or 0)
    attention_mode = str(getattr(args, "attention_mode", "ray_conditioned") or "ray_conditioned")
    disable_event_gate = bool(getattr(args, "disable_event_gate", False))

    model_kwargs: Dict[str, Any] = {"hidden_dim": int(args.hidden_dim)}
    if str(args.model_type).startswith("rind_"):
        model_kwargs["max_candidates"] = int(args.max_candidates)
        model_kwargs["reanchor_every"] = reanchor_every
        model_kwargs["use_static_mpnn"] = not bool(getattr(args, "disable_static_mpnn", False))
        model_kwargs["use_local_knn_readout"] = not bool(getattr(args, "disable_local_knn_readout", False))
        model_kwargs["use_fourier102"] = not bool(getattr(args, "disable_fourier102", False))
        model_kwargs["far_perspective"] = _profile_switch(
            args, "far_perspective", mainline and not getattr(args, "disable_fourier102", False)
        )
        model_kwargs["attention_mode"] = attention_mode
        model_kwargs.update(_candidate_attention_options(args))
        model_kwargs["disable_event_gate"] = disable_event_gate
    elif args.model_type in ("local_convgru_grid", "full_tile_convgru"):
        model_kwargs["crop_size"] = int(args.crop_size)
        model_kwargs["meters_per_pixel"] = float(first_shard.meters_per_pixel)
        model_kwargs["disable_event_gate"] = disable_event_gate
        if args.model_type == "local_convgru_grid":
            model_kwargs["spatial_execution_mode"] = convgru_mode

    model = build_model(args.model_type, **model_kwargs)
    model.num_azimuth_bins = int(args.num_azimuth_bins)
    days_per_batch = int(getattr(args, "days_per_batch", 1) or 1)
    rec_chunk = int(getattr(args, "receiver_chunk_size", 256) or 256)
    intra_frames = int(getattr(args, "max_intra_day_frames", 0) or 0)
    pure_sup = bool(obj_cfg.pure_supervised)
    label_budget_frac = float(obj_cfg.label_budget_fraction)
    matched_arm = getattr(args, "matched_objective", None)
    candidate_protocol = _candidate_protocol(args)

    ckpt_path = Path(args.checkpoint_out)
    collapse_jsonl = getattr(args, "collapse_jsonl", None)
    if collapse_jsonl is None:
        collapse_jsonl = ckpt_path.parent / f"{ckpt_path.stem}_collapse_diagnostics.jsonl"
    save_every_steps = int(getattr(args, "save_every_steps", 0) or 0)
    resume_from = getattr(args, "resume_from", None)
    resume_state_out = getattr(args, "resume_state_out", None)
    if resume_state_out is None and save_every_steps > 0:
        resume_state_out = ckpt_path.parent / f"{ckpt_path.stem}_resume_state.pt"
    max_train_steps = int(getattr(args, "max_train_steps", 0) or 0)

    objective_window = int(getattr(args, "objective_window_frames", 0))
    if objective_window < 0:
        raise ValueError("--objective-window-frames must be nonnegative")
    read_frames = math.gcd(intra_frames, objective_window) if objective_window else intra_frames

    logical_receivers = int(getattr(args,"objective_receivers",256)) if mainline else rec_chunk
    if mainline and logical_receivers <= 0:
        raise ValueError("--objective-receivers must be positive")

    def _epoch_batches(epoch_idx: int):
        rng = np.random.default_rng(int(args.seed) + epoch_idx * 1009)
        order = list(fit_tiles)
        rng.shuffle(order)
        for tid in order:
            sh = build_or_open_single_tile_shard(
                args.cache_dir,
                reader,
                tid,
                week_indices=week_indices,
                receiver_mode=args.receiver_mode,
                max_receivers=args.max_receivers,
                max_candidates=args.max_candidates,
                num_azimuth_bins=args.num_azimuth_bins,
                bbox_margin_m=args.bbox_margin_m,
            )
            batches = sh.iter_day_batches(
                days_per_batch=days_per_batch,
                receiver_chunk_size=logical_receivers,
                physical_receiver_read=rec_chunk,
                model_type=model.model_type,
                crop_size=int(getattr(args, "crop_size", 64)),
                shuffle_days=(read_frames <= 0),
                rng=rng,
                max_intra_day_frames=read_frames,
                convgru_execution_mode=convgru_mode,
            )

            if objective_window:
                from .fast_data import iter_objective_windows
                batches = iter_objective_windows(batches, objective_window)
            yield from batches

    train_stats = fit_model_fullday(
        model,
        _epoch_batches,
        firewall_token=firewall_token,
        expected_split_sha256=contract.split_sha256,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=dev,
        use_bf16_amp=(not args.disable_bf16),
        pin_and_async=True,
        prepare_gpu_tokens_fn=materialize_gpu_tokens_from_azimuth_bank,
        resource_guard=guard,
        seed=args.seed,
        objective_config=obj_cfg,
        cf_subgroup_size=int(getattr(args, "cf_subgroup_size", 0)),
        collapse_diagnostics_jsonl_path=collapse_jsonl,
        save_every_steps=save_every_steps,
        resume_state_out=resume_state_out,
        resume_from=resume_from,
        batch_schedule={"candidate_protocol": candidate_protocol, "objective_window_frames": objective_window,
                        "per_batch_frames": 0 if objective_window else intra_frames,
                        "receivers_per_update": logical_receivers, "days_per_batch": days_per_batch,
                        "fit_tiles": list(fit_tiles), "seed": int(args.seed),
                        "weeks": list(week_indices) if week_indices is not None else None,
                        "receiver_mode": str(args.receiver_mode), "max_receivers": args.max_receivers},
        max_train_steps=max_train_steps,
    )

    selection_rule = str(getattr(args, "selection_rule", "fixed_final_epoch"))
    save_prov = save_rind_checkpoint(
        ckpt_path,
        model,
        split_contract_sha256=contract.split_sha256,
        manifest_sha256=contract.manifest_sha256,
        fit_subset_name=args.fit_subset,
        fit_tile_ids=fit_tiles,
        trained_from_scratch_fit_only=True,
        seed=args.seed,
        calibrated=False,
        normalizer_tile_ids=model.consumed_normalizer_tile_ids,
        selection_rule=selection_rule,
        num_train_updates=int(train_stats["total_steps"]),
        trained_point_frames=int(train_stats["total_valid_point_frames"]),
        extra_metadata={
            "matched_objective": matched_arm,
            "candidate_protocol": candidate_protocol,
            "mainline": mainline,
            "receipt_class": receipt_class,
            "temporal_protocol": args.temporal_protocol,
            "receiver_mode": args.receiver_mode,
            "days_per_batch": days_per_batch,
            "receiver_chunk_size": rec_chunk,
            "max_intra_day_frames": intra_frames,
            "objective_window_frames": objective_window,
            "spatial_execution_mode": convgru_mode,
            "objective_config": train_stats["objective_config"],
            "cf_config": train_stats["cf_config"],
            "label_budget_fraction": label_budget_frac,
            "pure_supervised": pure_sup,
        },
    )

    model.eval()
    reloaded, _ = load_rind_checkpoint(
        ckpt_path,
        device=dev,
        expected_split_sha256=contract.split_sha256,
        expected_manifest_sha256=contract.manifest_sha256,
    )
    reloaded.eval()
    with torch.no_grad():
        sample_raw = next(
            first_shard.iter_day_batches(
                days_per_batch=1,
                receiver_chunk_size=min(rec_chunk, 64),
                model_type=model.model_type,
                crop_size=int(getattr(args, "crop_size", 64)),
                shuffle_days=False,
                convgru_execution_mode=convgru_mode,
                include_labels=False,
            )
        )
        sample_b = LabelReadForbiddenBatchView(
            materialize_gpu_tokens_from_azimuth_bank(move_batch_to_device(sample_raw, dev))
        )
        out_orig = model(sample_b)["logits"]
        out_reload = reloaded(sample_b)["logits"]
        max_reload_diff = float((out_orig - out_reload).abs().max().item())
        if max_reload_diff > 1e-6:
            raise RuntimeError(
                f"Post-save checkpoint reload parity failed: max_diff={max_reload_diff}"
            )

    ckpt_sha = compute_file_sha256(ckpt_path)
    receipt = {
        "benchmark_contract": BENCHMARK_CONTRACT_ID,
        "receipt_class": receipt_class,
        "model_type": model.model_type,
        "ablation_tier": model.ablation_tier,
        "input_tier": model.input_tier,
        "model_config": model.get_model_config(),
        "objective_config": train_stats["objective_config"],
        "cf_config": train_stats["cf_config"],
        "mainline": mainline,
        "fit_subset": args.fit_subset,
        "fit_tile_count": len(fit_tiles),
        "fit_tile_ids": fit_tiles,
        "consumed_fit_tile_ids": train_stats["consumed_fit_tile_ids"],
        "normalizer_tile_ids": model.consumed_normalizer_tile_ids,
        "blind_open_count": contract.blind_open_count,
        "temporal_protocol": args.temporal_protocol,
        "receiver_mode": args.receiver_mode,
        "seed": int(args.seed),
        "selection_rule": selection_rule,
        "checkpoint_path": str(ckpt_path.resolve()),
        "checkpoint_sha256": ckpt_sha,
        "post_save_reload_max_abs_diff": max_reload_diff,
        "provenance": save_prov,
        "objective_window_frames": objective_window,
        "objective_receiver_chunk_size": logical_receivers,
        "physical_receiver_read": rec_chunk,
        "physical_read_frames": read_frames,
        "objective_batch_mode": "fixed_window" if objective_window else "per_loader_batch",
        "training_summary": train_stats,
        "completed_at_utc": _utc_now_iso(),
    }
    if args.output:
        _write_json_atomic(args.output, receipt)
    if not args.quiet:
        print(json.dumps(receipt, indent=2))
    return receipt


def _candidate_protocol(args):
    from .ray_candidates import require_complete
    return dict(num_azimuth_bins=int(args.num_azimuth_bins), capacity=int(args.max_candidates),
                              horizon_m=float(args.bbox_margin_m), require_complete=require_complete.get(),
                              retrieval="source_ray_aabb_v1" if not args.num_azimuth_bins else "sector_range_v4")


def _verify_candidate_protocol(args, provenance):
    actual = _candidate_protocol(args)
    extra = provenance.get("extra") or {}
    expected = extra.get("calibration_candidate_protocol") or extra.get("candidate_protocol")
    if expected and not getattr(args,"allow_candidate_intervention",False):
        for key in ("capacity","horizon_m","retrieval","num_azimuth_bins"):
            if expected[key] != actual[key]:
                raise ValueError("Candidate protocol differs from checkpoint; use --allow-candidate-intervention for a paired comparison")
    return actual


def cmd_calibrate(args: argparse.Namespace) -> Dict[str, Any]:
    contract = _load_contract_from_args(args)
    receipt_class = _enforce_publication_or_diagnostic_mode(args, contract)
    dev = torch.device(args.device)
    guard = HostResourceGuard(device=dev)
    guard.check("cmd_calibrate:init")

    cal_tiles = contract.get_calibration_tiles()
    if args.max_tiles > 0:
        cal_tiles = cal_tiles[: int(args.max_tiles)]
    contract.assert_calibration_allowed(cal_tiles)

    model, prov = load_rind_checkpoint(
        args.checkpoint,
        device=dev,
        expected_split_sha256=contract.split_sha256,
        expected_manifest_sha256=contract.manifest_sha256,
    )
    actual_protocol = _verify_candidate_protocol(args,prov)
    if set(cal_tiles) & set(prov.get("fit_tile_ids") or []):
        raise RuntimeError("Calibration tiles overlap checkpoint `fit_tile_ids`!")

    reader = _configure_portable_reader(FullDayTileReader(contract, global_mesh_shared_dir=args.global_mesh_dir), args)
    week_indices = resolve_week_indices(args.temporal_protocol)
    days_per_batch = int(getattr(args, "days_per_batch", 2) or 2)
    rec_chunk = int(getattr(args, "receiver_chunk_size", 256) or 256)
    m_cfg = model.get_model_config()
    crop_sz = int(m_cfg.get("crop_size", 64))
    cal_convgru_mode = str(
        m_cfg.get(
            "spatial_execution_mode",
            "full_tile_512_halo" if model.model_type == "full_tile_convgru" else "per_receiver_crop",
        )
    )
    prov_extra = dict(prov.get("extra", {}) or {})
    cal_intra_frames = int(
        getattr(args, "max_intra_day_frames", 0) or prov_extra.get("max_intra_day_frames", 0) or 0
    )

    def _stream_cal_batches():
        for tid in cal_tiles:
            sh = build_or_open_single_tile_shard(
                args.cache_dir,
                reader,
                tid,
                week_indices=week_indices,
                receiver_mode=args.receiver_mode,
                max_receivers=args.max_receivers,
                max_candidates=args.max_candidates,
                num_azimuth_bins=args.num_azimuth_bins,
                bbox_margin_m=args.bbox_margin_m,
            )
            yield from sh.iter_day_batches(
                days_per_batch=days_per_batch,
                receiver_chunk_size=rec_chunk,
                model_type=model.model_type,
                crop_size=crop_sz,
                shuffle_days=False,
                max_intra_day_frames=cal_intra_frames,
                convgru_execution_mode=cal_convgru_mode,
            )

    cal_summary = calibrate_model_on_split(
        model,
        _stream_cal_batches(),
        contract=contract,
        device=dev,
        prepare_gpu_tokens_fn=materialize_gpu_tokens_from_azimuth_bank,
        resource_guard=guard,
    )

    out_ckpt = Path(args.checkpoint_out) if args.checkpoint_out else Path(args.checkpoint)
    cal_prov = save_rind_checkpoint(
        out_ckpt,
        model,
        split_contract_sha256=contract.split_sha256,
        manifest_sha256=contract.manifest_sha256,
        fit_subset_name=str(prov["fit_subset_name"]),
        fit_tile_ids=prov["fit_tile_ids"],
        trained_from_scratch_fit_only=bool(prov["trained_from_scratch_fit_only"]),
        seed=int(prov["seed"]),
        calibrated=True,
        calibration_tile_ids=cal_summary["calibration_tile_ids"],
        normalizer_tile_ids=prov.get("normalizer_tile_ids") or model.consumed_normalizer_tile_ids,
        selection_rule=str(prov.get("selection_rule", "fixed_final_epoch")),
        num_train_updates=int(prov.get("num_train_updates", 0)),
        trained_point_frames=int(prov.get("trained_point_frames", 0)),
        extra_metadata={**dict(prov.get("extra", {})), "calibration_candidate_protocol": actual_protocol, "calibration_summary": cal_summary},
    )

    receipt = {
        "benchmark_contract": BENCHMARK_CONTRACT_ID,
        "receipt_class": receipt_class,
        "model_type": model.model_type,
        "calibrated_checkpoint_path": str(out_ckpt.resolve()),
        "calibrated_checkpoint_sha256": compute_file_sha256(out_ckpt),
        "blind_open_count": contract.blind_open_count,
        "provenance": cal_prov,
        "calibration_summary": cal_summary,
        "completed_at_utc": _utc_now_iso(),
    }
    if args.output:
        _write_json_atomic(args.output, receipt)
    if not args.quiet:
        print(json.dumps(receipt, indent=2))
    return receipt


@torch.no_grad()
def _evaluate_single_tile_streaming(
    model: Any,
    shard: ContiguousTileMmapShard,
    dev: torch.device,
    *,
    guard: HostResourceGuard,
    days_per_batch: int = 2,
    receiver_chunk_size: int = 256,
    intra_day_frames: int = 0,
    temperature: float = 1.0,
    threshold: float = 0.5,
    eval_stride: int = 1,
    inference_mode: str = "recurrent",
    collect_logits_for_stratified: bool = False,
    collect_attention: bool = False,
) -> Tuple[ConfusionAccumulator, Dict[str, ConfusionAccumulator], Dict[str, ConfusionAccumulator], Dict[str, Any]]:
    """Stream day/receiver chunks for a single tile with bounded `O(T * chunk_N)` RAM (`< 48 MiB`).

    Accumulates `ConfusionAccumulator` (overall, solar regimes, boundary distance bands) and
    `long_day_diagnostics` strictly per receiver chunk `[r0, r1)`, NEVER allocating `(18808, N_total)`
    arrays on `all_valid` (`N ≈ 27,000`) tiles while remaining bit-for-bit identical to full-tile eval.
    """
    t_total = int(shard.num_frames)
    n_total = int(shard.num_receivers)
    m_cfg = model.get_model_config()
    crop_sz = int(m_cfg.get("crop_size", 64))
    convgru_mode = str(
        m_cfg.get(
            "spatial_execution_mode",
            "full_tile_512_halo" if model.model_type == "full_tile_convgru" else "per_receiver_crop",
        )
    )
    stride_eff = max(1, int(eval_stride))
    eff_inf_mode = str(inference_mode or "recurrent")
    if stride_eff > 1 and eff_inf_mode == "recurrent":
        eff_inf_mode = "causal_rollout"
    candidate_stride = (stride_eff if hasattr(model, "evaluate_strided_rollout")
                        and eff_inf_mode != "static_direct" else 1)

    day_ids = shard._mmap_slice_copy("day_ids")
    ts_s = shard._mmap_slice_copy("timestamps_unix_s")
    days_str = [str(d) for d in day_ids]

    # Precompute long-day spans once (`O(T)` scalars) so per-receiver-chunk lag stats merge identically
    spans: List[Tuple[int, int]] = []
    if t_total > 0:
        s0 = 0
        for idx_t in range(1, t_total):
            if days_str[idx_t] != days_str[s0]:
                spans.append((s0, idx_t))
                s0 = idx_t
        spans.append((s0, t_total))
    max_day_len = max(((b - a) for a, b in spans), default=0)
    long_spans = [(a, b) for a, b in spans if (b - a) >= 400]
    if not long_spans and spans:
        long_spans = [(a, b) for a, b in spans if (b - a) == max_day_len]

    t_acc = ConfusionAccumulator()
    t_regs = {r: ConfusionAccumulator() for r in SOLAR_REGIME_ORDER}
    t_bands = {b: ConfusionAccumulator() for b in DISTANCE_BAND_ORDER}

    early_correct = 0
    early_total = 0
    late_correct = 0
    late_total = 0
    lag_step_sum = 0.0
    lag_histogram = {}
    lag_step_count = 0
    lag_counts = dict(true_events=0,predicted_events=0,matched_events=0,missed_events=0,spurious_events=0)

    max_latent = 0.0
    extra_nf = 0
    nonfinite_windows = 0
    import shutil
    if collect_attention and "scene_triangles_xyz" in shard.array_paths:
        k = shard.provenance["max_candidates"]
        required = t_total*n_total*(9*k + (4*k if candidate_stride == 1 else 0) + 4)
        free = shutil.disk_usage(tempfile.gettempdir()).free
        if required > free:
            raise OSError(f"Blocker-bucket arrays require {required} scratch bytes; {free} free")
    static_reuses = 0
    frozen_static = None
    collected_ray_ids = collected_ray_mask = None
    ray_budget = {}
    actual_anchor_frames = 0
    candidate_total_frames = 0
    collected_attention = None
    collected_logits_tn = None
    if collect_logits_for_stratified and t_total and n_total:
        with tempfile.TemporaryFile() as backing:
            collected_logits_tn = np.memmap(backing, dtype=np.float32, mode="w+", shape=(t_total, n_total))

    if t_total == 0 or n_total == 0:
        lag_diag = compute_long_day_lag_diagnostics(
            np.zeros((t_total, 0), dtype=np.float64),
            np.zeros((t_total, 0), dtype=np.int8),
            np.zeros((t_total, 0), dtype=bool),
            day_ids,
            threshold=threshold,
        )
        return t_acc, t_regs, t_bands, {
            "long_day_diagnostics": lag_diag,
            "sampling_disclosure": shard.receiver_sampling_disclosure(),
            "inference_mode": eff_inf_mode,
        }

    # For non-full-grid models, cap forward chunk size to 256 when `<= 0` so per-step GPU/CPU buffers stay bounded;
    # for `full_tile_512_halo` ConvGRU, `iter_day_batches` automatically runs 1 full-grid pass/day.
    is_full_grid_convgru = (
        model.model_type in ("local_convgru_grid", "full_tile_convgru")
        and convgru_mode == "full_tile_512_halo"
    )
    fwd_chunk_size = (
        n_total
        if is_full_grid_convgru
        else (int(receiver_chunk_size) if int(receiver_chunk_size) > 0 else 256)
    )

    def _reduce_receiver_block(
        r0_blk: int,
        r1_blk: int,
        lg_blk: np.ndarray,
        hc_blk_u8: np.ndarray,
    ) -> None:
        nonlocal early_correct, early_total, late_correct, late_total, lag_step_sum, lag_step_count
        blk_sl = slice(r0_blk, r1_blk)
        truth_blk = shard._mmap_slice_copy("truth", (slice(None), blk_sl))
        valid_blk = shard._mmap_slice_copy("valid_mask", (slice(None), blk_sl))
        bdist_blk = shard._mmap_slice_copy("boundary_distance_m", (slice(None), blk_sl))
        bval_blk = shard._mmap_slice_copy("boundary_distance_valid", (slice(None), blk_sl))

        sub_step = 1024
        blk_w = r1_blk - r0_blk
        for sub0 in range(0, blk_w, sub_step):
            sub1 = min(blk_w, sub0 + sub_step)
            s_sl = slice(sub0, sub1)
            lg_c = np.asarray(lg_blk[:, s_sl], dtype=np.float32)
            hc_c = np.asarray(hc_blk_u8[:, s_sl], dtype=np.uint8) > 0
            truth_c = truth_blk[:, s_sl]
            valid_c = valid_blk[:, s_sl]
            bdist_c = bdist_blk[:, s_sl]
            bval_c = bval_blk[:, s_sl]

            c_acc, c_regs, c_bands, _ = evaluate_trajectory_predictions(
                lg_c,
                truth_c,
                valid_c,
                solar_regimes=shard.solar_regimes,
                day_ids=day_ids,
                timestamps_unix_s=ts_s,
                has_active_candidates=hc_c,
                boundary_distance_m=bdist_c,
                boundary_distance_valid=bval_c,
                return_distance_bands=True,
                decision_threshold=threshold,
                temperature=temperature,
                max_latent_norm=0.0,
                extra_nonfinite_count=0,
            )
            t_acc.merge(c_acc)
            for rk in SOLAR_REGIME_ORDER:
                t_regs[rk].merge(c_regs[rk])
            for bk in DISTANCE_BAND_ORDER:
                t_bands[bk].merge(c_bands[bk])

            probs_c = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(lg_c.astype(np.float64) / max(temperature, 1e-4), -60.0, 60.0)
                )
            )
            val_eff = np.asarray(valid_c, dtype=bool) & (np.asarray(truth_c, dtype=np.int8) >= 0)
            pred_bin = probs_c >= float(threshold)
            gt_bin = np.asarray(truth_c, dtype=np.int8) == 1
            for sa, sb in long_spans:
                d_len = sb - sa
                if d_len < 2:
                    continue
                mid = sa + d_len // 2
                v_e = val_eff[sa:mid]
                if np.any(v_e):
                    early_correct += int(np.sum(pred_bin[sa:mid][v_e] == gt_bin[sa:mid][v_e]))
                    early_total += int(np.sum(v_e))
                v_l = val_eff[mid:sb]
                if np.any(v_l):
                    late_correct += int(np.sum(pred_bin[mid:sb][v_l] == gt_bin[mid:sb][v_l]))
                    late_total += int(np.sum(v_l))
                counts, seconds = directed_event_matches(gt_bin[sa:sb],pred_bin[sa:sb],val_eff[sa:sb],ts_s[sa:sb])
                for key,value in counts.items():
                    lag_counts[key] += value
                for lag in seconds:
                    key = str(int(lag))
                    lag_histogram[key] = lag_histogram.get(key, 0) + 1
                lag_step_sum += sum(map(abs, seconds))/120.
                lag_step_count += len(seconds)

    with tempfile.TemporaryDirectory(prefix="rind_eval_stream_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        logits_npy_path = tmp_dir / "eval_logits_f32.npy"
        cands_npy_path = tmp_dir / "eval_has_cands_u8.npy"
        lg_fh = None
        hc_fh = None
        if is_full_grid_convgru:
            lg_fh = logits_npy_path.open("wb")
            lg_fh.write(_write_npy_header_bytes((t_total, n_total), np.float32))
            hc_fh = cands_npy_path.open("wb")
            hc_fh.write(_write_npy_header_bytes((t_total, n_total), np.uint8))

        try:
            active_range: Optional[Tuple[int, int]] = None
            super_r0: Optional[int] = None
            super_r1: int = 0
            super_lg: Optional[np.ndarray] = None
            super_hc: Optional[np.ndarray] = None
            super_max_cols = 8192
            prev_latent: Optional[torch.Tensor] = None
            prev_conv: Optional[torch.Tensor] = None
            anchor_state = None
            prior_ray_ids = prior_ray_mask = None
            step_i = 0
            frame_cursors: Dict[Tuple[int, int], int] = {}

            for raw_b in shard.iter_day_batches(
                days_per_batch=days_per_batch,
                receiver_chunk_size=fwd_chunk_size,
                model_type=model.model_type,
                crop_size=crop_sz,
                shuffle_days=False,
                max_intra_day_frames=0 if eff_inf_mode == "offline_interp" else intra_day_frames,
                convgru_execution_mode=convgru_mode,
                include_labels=False,
                candidate_stride=candidate_stride,
                candidate_mode=eff_inf_mode,
            ):
                if step_i % 8 == 0:
                    guard.check(f"eval_stream:{shard.tile_id}:s{step_i}")
                step_i += 1

                r0, r1 = raw_b["receiver_range"]
                if active_range != (r0, r1):
                    active_range = (r0, r1)
                    frozen_static = None
                    prev_latent = None
                    prev_conv = None
                    anchor_state = None
                    prior_ray_ids = prior_ray_mask = None
                    if not is_full_grid_convgru:
                        if super_r0 is None:
                            cap = min(max(super_max_cols, r1 - r0), n_total - r0)
                            super_lg = np.empty((t_total, cap), dtype=np.float32)
                            super_hc = np.empty((t_total, cap), dtype=np.uint8)
                            super_r0 = r0
                            super_r1 = r0
                        elif (
                            super_lg is None
                            or super_hc is None
                            or r0 != super_r1
                            or (r1 - super_r0) > super_lg.shape[1]
                        ):
                            assert super_lg is not None and super_hc is not None
                            if super_r1 > super_r0:
                                _reduce_receiver_block(
                                    super_r0,
                                    super_r1,
                                    super_lg[:, : (super_r1 - super_r0)],
                                    super_hc[:, : (super_r1 - super_r0)],
                                )
                            need_cap = min(max(super_max_cols, r1 - r0), n_total - r0)
                            if super_lg.shape[1] < need_cap:
                                super_lg = np.empty((t_total, need_cap), dtype=np.float32)
                                super_hc = np.empty((t_total, need_cap), dtype=np.uint8)
                            super_r0 = r0
                            super_r1 = r0
                        super_r1 = r1

                if "frame_range" in raw_b:
                    f0, f1 = raw_b["frame_range"]
                else:
                    cur_f = frame_cursors.get((r0, r1), 0)
                    f_len = int(raw_b["sun_dirs"].shape[0])
                    f0, f1 = cur_f, cur_f + f_len
                    frame_cursors[(r0, r1)] = f1

                if "ray_candidate_count" in raw_b:
                    from .ray_candidates import record_budget
                    k = raw_b["az_mask"].shape[-1]
                    record_budget(ray_budget, raw_b["ray_candidate_count"], k)
                    actual_anchor_frames += int(raw_b.get("candidate_query_frames", f1 - f0))
                    candidate_total_frames += f1 - f0
                    if collect_attention:
                        if collected_ray_ids is None:
                            with tempfile.TemporaryFile() as f_ids, tempfile.TemporaryFile() as f_mask:
                                shape = (t_total, n_total, k)
                                collected_ray_ids = np.memmap(f_ids, np.int64, mode="w+", shape=shape)
                                collected_ray_mask = np.memmap(f_mask, bool, mode="w+", shape=shape)
                        bank_idx = raw_b["azimuth_bin_idx"].numpy()
                        ids = raw_b["az_source_ids"].transpose(0, 1).numpy()[bank_idx].copy()
                        masks = raw_b["az_mask"].transpose(0, 1).numpy()[bank_idx].copy()
                        anchor_mask = raw_b.get("candidate_anchor_mask")
                        if anchor_mask is not None:
                            anchor_indices = np.flatnonzero(np.asarray(anchor_mask))
                            prefix_end = int(anchor_indices[0]) if anchor_indices.size else f1 - f0
                            if prefix_end and bool(raw_b.get("carry_state_from_prev_chunk", False)):
                                if prior_ray_ids is None:
                                    raise RuntimeError("Missing previous candidate IDs for strided continuation")
                                ids[:prefix_end] = prior_ray_ids
                                masks[:prefix_end] = prior_ray_mask
                        collected_ray_ids[f0:f1,r0:r1] = ids
                        collected_ray_mask[f0:f1,r0:r1] = masks
                        prior_ray_ids, prior_ray_mask = ids[-1].copy(), masks[-1].copy()

                b_dev = move_batch_to_device(
                    LabelReadForbiddenBatchView(raw_b), dev, pin_and_async=False
                )
                if frozen_static is not None:
                    b_dev["frozen_static_geometry"] = frozen_static
                    static_reuses += 1
                if bool(b_dev.get("carry_state_from_prev_chunk", False)) and prev_latent is not None:
                    b_dev["initial_latent_state"] = prev_latent
                    b_dev["stride_anchor_state"] = anchor_state
                    if prev_conv is not None:
                        b_dev["initial_conv_state"] = prev_conv
                else:
                    prev_latent = None
                    prev_conv = None

                if collect_attention and candidate_stride == 1:
                    b_dev["return_candidate_attention"] = True

                if stride_eff > 1 and hasattr(model, "evaluate_strided_rollout"):
                    # Do NOT materialize full-T tokens eagerly; let `evaluate_strided_rollout` build anchors lazily!
                    roll_mode = (
                        "offline_interp"
                        if eff_inf_mode == "offline_interp"
                        else ("static_direct" if eff_inf_mode == "static_direct" else "causal_rollout")
                    )
                    out = model.evaluate_strided_rollout(
                        LabelReadForbiddenBatchView(b_dev),
                        stride=stride_eff,
                        mode=roll_mode,
                        use_calibrated_temperature=False,
                    )
                elif eff_inf_mode != "recurrent" and hasattr(model, "forward_with_mode"):
                    b_dev = materialize_gpu_tokens_from_azimuth_bank(b_dev)
                    out = model.forward_with_mode(
                        LabelReadForbiddenBatchView(b_dev),
                        inference_mode=eff_inf_mode,
                        stride=stride_eff,
                        use_calibrated_temperature=False,
                    )
                else:
                    b_dev = materialize_gpu_tokens_from_azimuth_bank(b_dev)
                    out = model(
                        LabelReadForbiddenBatchView(b_dev),
                        use_calibrated_temperature=False,
                    )

                if not model.training and out.get("static_geom_memory") is not None:
                    frozen_static = out["static_geom_memory"].detach()
                anchor_state = out.get("stride_anchor_state")
                if "final_latent_state" in out and out["final_latent_state"] is not None:
                    prev_latent = out["final_latent_state"].detach()
                if "final_conv_state" in out and out["final_conv_state"] is not None:
                    prev_conv = out["final_conv_state"].detach()

                lg_np = np.ascontiguousarray(
                    out["logits"].detach().cpu().numpy(), dtype=np.float32
                )
                if lg_np.ndim == 3:
                    lg_np = lg_np.squeeze(0)
                if collected_logits_tn is not None:
                    collected_logits_tn[f0:f1, r0:r1] = lg_np

                weights = out.get("candidate_attn_weights")
                if collect_attention and candidate_stride == 1 and weights is not None:
                    weights_np = weights.detach().float().cpu().numpy().reshape(f1-f0, r1-r0, -1)
                    if collected_attention is None:
                        with tempfile.TemporaryFile(dir=tmp_dir_str) as backing:
                            collected_attention = np.memmap(
                                backing, dtype=np.float32, mode="w+",
                                shape=(t_total, n_total, weights_np.shape[-1]),
                            )
                    collected_attention[f0:f1, r0:r1] = weights_np

                if "has_active_candidates" in out and out["has_active_candidates"] is not None:
                    hc_np = np.ascontiguousarray(
                        out["has_active_candidates"].detach().cpu().numpy(), dtype=np.uint8
                    )
                    if hc_np.ndim == 3:
                        hc_np = hc_np.squeeze(0)
                elif "candidate_mask" in b_dev:
                    cm_np = b_dev["candidate_mask"].detach().cpu().numpy().astype(bool)
                    if cm_np.ndim == 4:
                        cm_np = cm_np.squeeze(0)
                    hc_np = np.ascontiguousarray(np.any(cm_np, axis=-1), dtype=np.uint8)
                elif "az_mask" in raw_b:
                    az_m = raw_b["az_mask"].numpy().astype(bool)
                    az_b = raw_b["azimuth_bin_idx"].numpy().astype(np.int64)
                    hc_np = np.ascontiguousarray(
                        np.any(np.transpose(az_m, (1, 0, 2))[az_b], axis=-1), dtype=np.uint8
                    )
                else:
                    az_m = shard._mmap_slice_copy("az_mask", slice(r0, r1)).astype(bool)
                    az_b = raw_b["azimuth_bin_idx"].numpy().astype(np.int64)
                    hc_np = np.ascontiguousarray(
                        np.any(np.transpose(az_m, (1, 0, 2))[az_b], axis=-1), dtype=np.uint8
                    )

                if is_full_grid_convgru:
                    assert lg_fh is not None and hc_fh is not None
                    lg_fh.write(lg_np.tobytes(order="C"))
                    hc_fh.write(hc_np.tobytes(order="C"))
                else:
                    assert super_lg is not None and super_hc is not None and super_r0 is not None
                    c0 = r0 - super_r0
                    c1 = r1 - super_r0
                    super_lg[f0:f1, c0:c1] = lg_np
                    super_hc[f0:f1, c0:c1] = hc_np

                lat_n = float(out["max_latent_norm"])
                max_latent = max(max_latent, lat_n)
                nf_c = int(out["nonfinite_count"])
                if nf_c:
                    raise FloatingPointError("non-finite model outputs in streaming evaluation")
                extra_nf += nf_c
                if nf_c > 0:
                    nonfinite_windows += 1

            if is_full_grid_convgru:
                if lg_fh is not None:
                    lg_fh.close()
                    lg_fh = None
                if hc_fh is not None:
                    hc_fh.close()
                    hc_fh = None
                conv_reduce_step = 8192
                for mr0 in range(0, n_total, conv_reduce_step):
                    mr1 = min(n_total, mr0 + conv_reduce_step)
                    r_sl = slice(mr0, mr1)
                    lg_blk = _read_npy_slice_no_mmap(logits_npy_path, (slice(None), r_sl))
                    hc_blk = _read_npy_slice_no_mmap(cands_npy_path, (slice(None), r_sl))
                    _reduce_receiver_block(mr0, mr1, lg_blk, hc_blk)
            elif (
                super_r0 is not None
                and super_r1 > super_r0
                and super_lg is not None
                and super_hc is not None
            ):
                _reduce_receiver_block(
                    super_r0,
                    super_r1,
                    super_lg[:, : (super_r1 - super_r0)],
                    super_hc[:, : (super_r1 - super_r0)],
                )
        finally:
            if lg_fh is not None:
                lg_fh.close()
            if hc_fh is not None:
                hc_fh.close()

    t_acc.nonfinite_count += extra_nf
    t_acc.nonfinite_window_count += nonfinite_windows
    t_acc.max_latent_norm = max(t_acc.max_latent_norm, max_latent)

    early_acc = (float(early_correct) / float(early_total)) if early_total > 0 else None
    late_acc = (float(late_correct) / float(late_total)) if late_total > 0 else None
    drift = (late_acc - early_acc) if (early_acc is not None and late_acc is not None) else None
    mean_lag = (lag_step_sum / float(lag_step_count)) if lag_step_count > 0 else None
    lag_diag = {
        **lag_counts,
        "signed_transition_lag_seconds_histogram": lag_histogram,
        "long_day_count": len(long_spans),
        "max_day_frames": int(max_day_len),
        "early_half_accuracy": early_acc,
        "late_half_accuracy": late_acc,
        "late_minus_early_accuracy_drift": drift,
        "mean_transition_step_lag_frames": mean_lag,
        "mean_transition_lag_seconds": (mean_lag * 120.0) if mean_lag is not None else None,
    }
    extra_info = {
        "long_day_diagnostics": lag_diag,
        "sampling_disclosure": shard.receiver_sampling_disclosure(),
        "eval_stride": stride_eff,
        "inference_mode": eff_inf_mode,
        "is_causal": eff_inf_mode != "offline_interp",
        "collected_logits": collected_logits_tn,
        "collected_attention": collected_attention,
        "collected_ray_ids": collected_ray_ids,
        "collected_ray_mask": collected_ray_mask,
        "ray_budget": ray_budget,
        "actual_anchor_frames": actual_anchor_frames,
        "candidate_total_frames": candidate_total_frames,
        "bvh_savings_rate": (1.0 - actual_anchor_frames / candidate_total_frames
                             if candidate_total_frames else 0.0),
        "static_geometry_batch_reuses": static_reuses,
    }
    return t_acc, t_regs, t_bands, extra_info


@torch.no_grad()
def _run_bucket_evaluation(
    args: argparse.Namespace,
    contract: SplitContract,
    bucket_name: str,
    *,
    unseal_freeze_manifest: Optional[Mapping[str, Any]] = None,
    return_tile_accumulators: bool = False,
) -> Dict[str, Any] | Tuple[Dict[str, Any], Dict[str, ConfusionAccumulator]]:
    receipt_class = _enforce_publication_or_diagnostic_mode(args, contract)
    dev = torch.device(args.device)
    guard = HostResourceGuard(device=dev)
    guard.check(f"eval:{bucket_name}:init")

    require_cal = not bool(getattr(args, "allow_uncalibrated", False))
    model, prov = load_rind_checkpoint(
        args.checkpoint,
        device=dev,
        expected_split_sha256=contract.split_sha256,
        expected_manifest_sha256=contract.manifest_sha256,
        require_calibrated=require_cal,
    )
    actual_protocol = _verify_candidate_protocol(args,prov)
    model.eval()

    reader = _configure_portable_reader(FullDayTileReader(contract, global_mesh_shared_dir=args.global_mesh_dir), args)
    week_indices = resolve_week_indices(args.temporal_protocol)

    tile_ids = contract.get_eval_tiles(
        bucket_name,
        model_provenance=prov,
        allow_empty=True,
        unseal_freeze_manifest=unseal_freeze_manifest,
    )
    if getattr(args, "max_tiles", 0) and int(args.max_tiles) > 0:
        tile_ids = tile_ids[: int(args.max_tiles)]

    mainline = _profile_switch(args, "mainline", bool((prov.get("extra") or {}).get("mainline", False)))
    want_blocker_buckets = _profile_switch(args, "blocker_distance_buckets", mainline)
    want_contours = _profile_switch(args, "shadow_contours", mainline)
    if getattr(reader, "portable_index", None) is not None and (want_blocker_buckets or want_contours):
        raise RuntimeError("Portable mode does not support raw-scene blocker/contour diagnostics; disable both explicitly")
    diagnostics = {"mainline": mainline, "blocker_distance_buckets": want_blocker_buckets,
                   "shadow_contours": want_contours}
    temp = float(model.calibrated_temperature.item())
    thresh = float(model.calibrated_threshold.item())
    ckpt_sha = compute_file_sha256(Path(args.checkpoint))

    if not tile_ids:
        bucket_payload = {
            "status": "NO_BLIND_TILES_IN_SNAPSHOT_V1" if bucket_name == "test_spatial_unexposed" else "EMPTY_BUCKET",
            "bucket_name": bucket_name,
            "tile_count": 0,
            "state_accuracy": None,
            "formatted_summary": "NA",
            "note": f"N=0 ready tiles currently registered for {bucket_name}; metric values are NA.",
        }
        receipt = {
            "benchmark_contract": BENCHMARK_CONTRACT_ID,
            "receipt_class": receipt_class,
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": ckpt_sha,
            "diagnostics": diagnostics,
            "model_type": model.model_type,
            "ablation_tier": model.ablation_tier,
            "input_tier": model.input_tier,
            "blind_open_count": contract.blind_open_count,
            "evaluation_buckets_strictly_separated": {bucket_name: bucket_payload},
            "completed_at_utc": _utc_now_iso(),
        }
        if args.output:
            _write_json_atomic(args.output, receipt)
        if not args.quiet:
            print(json.dumps(receipt, indent=2))
        if return_tile_accumulators:
            return receipt, {}
        return receipt

    contract.assert_eval_bucket_allowed(
        bucket_name,
        tile_ids,
        model_provenance=prov,
        require_calibrated_provenance=require_cal,
        unseal_freeze_manifest=unseal_freeze_manifest,
        check_one_shot=False,
    )

    overall_acc = ConfusionAccumulator()
    regime_accs = {r: ConfusionAccumulator() for r in SOLAR_REGIME_ORDER}
    band_accs = {b: ConfusionAccumulator() for b in DISTANCE_BAND_ORDER}
    per_tile_accs: List[ConfusionAccumulator] = []
    per_tile_map: Dict[str, ConfusionAccumulator] = {}
    spatial_groups: List[str] = []
    per_tile_rows: List[Dict[str, Any]] = []

    days_per_batch = int(getattr(args, "days_per_batch", 2) or 2)
    rec_chunk = int(getattr(args, "receiver_chunk_size", 256) or 256)
    prov_extra = dict(prov.get("extra", {}) or {})
    intra_frames = int(
        getattr(args, "max_intra_day_frames", 0) or prov_extra.get("max_intra_day_frames", 0) or 0
    )
    eval_stride = int(getattr(args, "eval_stride", 1) or 1)
    inference_mode = str(getattr(args, "inference_mode", "recurrent") or "recurrent")

    for tid in tile_ids:
        shard = build_or_open_single_tile_shard(
            args.cache_dir,
            reader,
            tid,
            week_indices=week_indices,
            receiver_mode=getattr(args, "receiver_mode", "sampled"),
            max_receivers=args.max_receivers,
            max_candidates=args.max_candidates,
            num_azimuth_bins=args.num_azimuth_bins,
            bbox_margin_m=getattr(args, "bbox_margin_m", DEFAULT_SCENE_HORIZON_M),
        )
        t_acc, t_regs, t_bands, extra_info = _evaluate_single_tile_streaming(
            model,
            shard,
            dev,
            guard=guard,
            days_per_batch=days_per_batch,
            receiver_chunk_size=rec_chunk,
            intra_day_frames=intra_frames,
            temperature=temp,
            threshold=thresh,
            eval_stride=eval_stride,
            inference_mode=inference_mode,
            collect_logits_for_stratified=want_blocker_buckets or want_contours,
            collect_attention=want_blocker_buckets,
        )
        if t_acc.nonfinite_count > 0:
            raise RuntimeError(
                f"Non-finite rollout values detected on {tid} in {bucket_name}: {t_acc.nonfinite_count}"
            )
        overall_acc.merge(t_acc)
        for r_k in SOLAR_REGIME_ORDER:
            regime_accs[r_k].merge(t_regs[r_k])
        for b_k in DISTANCE_BAND_ORDER:
            band_accs[b_k].merge(t_bands[b_k])
        per_tile_accs.append(t_acc)
        per_tile_map[tid] = t_acc
        spatial_groups.append(shard.spatial_group)
        blocker_strat: Optional[Dict[str, Any]] = None
        if (want_blocker_buckets or want_contours) and extra_info.get("collected_logits") is not None:
            geom_traj = reader.load_tile_fullday(
                tid,
                week_indices=week_indices[:1],
                receiver_mode=getattr(args, "receiver_mode", "sampled"),
                max_receivers=args.max_receivers,
                max_candidates=args.max_candidates,
                num_azimuth_bins=args.num_azimuth_bins,
                bbox_margin_m=getattr(args, "bbox_margin_m", DEFAULT_SCENE_HORIZON_M),
                include_labels=False,
            )
            if not np.array_equal(geom_traj.receiver_rc, shard._mmap_slice_copy("receiver_rc")):
                raise ValueError("geometry and predictions use different receiver ordering")
        if want_blocker_buckets and extra_info.get("collected_logits") is not None:
            blocker_strat = compute_blocker_distance_stratified_metrics(
                extra_info["collected_logits"],
                np.load(shard.array_paths["truth"], mmap_mode="r"),
                np.load(shard.array_paths["valid_mask"], mmap_mode="r"),
                receivers_xyz=shard._mmap_slice_copy("receivers_xyz"),
                sun_directions_xyz=shard._mmap_slice_copy("sun_dirs"),
                scene_triangles_xyz=geom_traj.scene_triangles_xyz,
                scene_opaque_mask=geom_traj.scene_opaque,
                scene_source_triangle_ids=geom_traj.scene_source_ids,
                exclude_source_triangle_ids=geom_traj.receiver_source_triangle_ids,
                day_ids=shard._mmap_slice_copy("day_ids"),
                timestamps_unix_s=shard._mmap_slice_copy("timestamps_unix_s"),
                candidate_source_triangle_ids=(extra_info.get("collected_ray_ids") if geom_traj.ray_candidates else shard._mmap_slice_copy("az_source_ids")),
                candidate_mask=(extra_info.get("collected_ray_mask") if geom_traj.ray_candidates else shard._mmap_slice_copy("az_mask")),
                candidate_layout="time" if geom_traj.ray_candidates else "azimuth_bank",
                candidate_attn_weights=extra_info.get("collected_attention"),
                decision_threshold=thresh,
                temperature=temp,
                t_max_m=float(getattr(args, "bbox_margin_m", DEFAULT_SCENE_HORIZON_M)),
            )
        tile_row: Dict[str, Any] = {
            "tile_id": tid,
            "spatial_group": shard.spatial_group,
            "source_batch": contract.tiles[tid].source_batch,
            "historical_exposure_status": contract.tiles[tid].historical_exposure_status,
            "ray_budget": extra_info.get("ray_budget"),
            "actual_anchor_frames": extra_info.get("actual_anchor_frames"),
            "candidate_total_frames": extra_info.get("candidate_total_frames"),
            "bvh_savings_rate": extra_info.get("bvh_savings_rate"),
            "static_geometry_batch_reuses": extra_info.get("static_geometry_batch_reuses"),
            "sampling_disclosure": extra_info["sampling_disclosure"],
            "inference_mode": extra_info["inference_mode"],
            "is_causal": extra_info["is_causal"],
            "long_day_diagnostics": extra_info["long_day_diagnostics"],
            "distance_bands": {b_k: t_bands[b_k].summarize() for b_k in DISTANCE_BAND_ORDER},
            "summary": t_acc.summarize(),
        }
        if want_contours and extra_info.get("collected_logits") is not None:
            tile_row["shadow_contours"] = compute_shadow_contour_f1(
                extra_info["collected_logits"], np.load(shard.array_paths["truth"], mmap_mode="r"),
                np.load(shard.array_paths["valid_mask"], mmap_mode="r"), shard._mmap_slice_copy("receiver_rc"),
                geom_traj.world_x_m, geom_traj.world_z_m, temperature=temp, decision_threshold=thresh,
            )
        if blocker_strat is not None:
            tile_row["blocker_distance_buckets"] = blocker_strat
        per_tile_rows.append(tile_row)

    boot_ci = bootstrap_tile_ci_95(per_tile_accs, spatial_group_ids=spatial_groups)
    bucket_report = {
        "candidate_protocol": actual_protocol,
        "status": "EVALUATED",
        "bucket_name": bucket_name,
        "tile_count": len(tile_ids),
        "spatial_group_count": len(set(spatial_groups)),
        "temporal_protocol": args.temporal_protocol,
        "receiver_mode": getattr(args, "receiver_mode", "sampled"),
        "calibrated_temperature": temp,
        "calibrated_threshold": thresh,
        "overall": overall_acc.summarize(),
        "tile_bootstrap_ci95": boot_ci,
        "solar_regimes": {r_k: regime_accs[r_k].summarize() for r_k in SOLAR_REGIME_ORDER},
        "boundary_distance_bands": {b_k: band_accs[b_k].summarize() for b_k in DISTANCE_BAND_ORDER},
        "per_tile": per_tile_rows,
    }

    ledger_summary: Optional[Dict[str, Any]] = None
    if bucket_name == "val_spatial_clean":
        ledger_path = (
            Path(args.ledger_file)
            if getattr(args, "ledger_file", None)
            else (Path(args.output).parent / "val_exposure_ledger.json" if args.output else Path(args.cache_dir) / "val_exposure_ledger.json")
        )
        ledger_doc = _append_val_exposure_ledger(
            ledger_path,
            {
                "timestamp_utc": _utc_now_iso(),
                "receipt_class": receipt_class,
                "model_type": model.model_type,
                "checkpoint_sha256": ckpt_sha,
                "temporal_protocol": args.temporal_protocol,
                "tile_count": len(tile_ids),
                "blind_open_count": contract.blind_open_count,
                "state_accuracy": bucket_report["overall"]["state_accuracy"],
            },
        )
        ledger_summary = {
            "ledger_path": str(ledger_path.resolve()),
            "total_val_evaluations": ledger_doc["total_val_evaluations"],
        }

    guard.check(f"eval:{bucket_name}:done")
    receipt = {
        "benchmark_contract": BENCHMARK_CONTRACT_ID,
        "receipt_class": receipt_class,
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": ckpt_sha,
        "diagnostics": diagnostics,
        "model_type": model.model_type,
        "ablation_tier": model.ablation_tier,
        "input_tier": model.input_tier,
        "model_config": model.get_model_config(),
        "head_parameter_norms": compute_head_parameter_norms(model),
        "blind_open_count": contract.blind_open_count,
        "val_exposure_ledger": ledger_summary,
        "provenance": prov,
        "resource_telemetry": guard.summary(),
        "evaluation_buckets_strictly_separated": {bucket_name: bucket_report},
        "completed_at_utc": _utc_now_iso(),
    }
    if args.output:
        _write_json_atomic(args.output, receipt)
    if not args.quiet:
        print(json.dumps(receipt, indent=2))
    if return_tile_accumulators:
        return receipt, per_tile_map
    return receipt


def cmd_eval_fullday(args: argparse.Namespace) -> Dict[str, Any]:
    contract = _load_contract_from_args(args)
    b_name = str(getattr(args, "bucket", "val_spatial_clean"))
    if b_name in ("all_separated", "dev_separated"):
        b_name = "val_spatial_clean"
    if b_name == "test_spatial_unexposed":
        raise RuntimeError(
            "BLIND TEST LOCK: `eval` / `eval-fullday` cannot open `test_spatial_unexposed`! "
            "Use `unseal-blind-test --freeze-manifest <path>` ONLY once after freezing all checkpoints."
        )
    if b_name != "val_spatial_clean" and b_name != "test_spatial_legacy_reassigned":
        raise ValueError(f"Unsupported evaluation bucket {b_name!r}")
    return _run_bucket_evaluation(args, contract, b_name)


def cmd_eval_legacy(args: argparse.Namespace) -> Dict[str, Any]:
    contract = _load_contract_from_args(args)
    return _run_bucket_evaluation(args, contract, "test_spatial_legacy_reassigned")


def cmd_unseal_blind_test(args: argparse.Namespace) -> Dict[str, Any]:
    contract = _load_contract_from_args(args)
    freeze_path = Path(args.freeze_manifest)
    if not freeze_path.exists():
        raise FileNotFoundError(f"Blind-test freeze manifest not found: {freeze_path}")
    freeze_doc = json.loads(freeze_path.read_text(encoding="utf-8"))
    contract.verify_blind_unseal_manifest(freeze_doc, one_shot_only=True)

    ckpt_sha = compute_file_sha256(Path(args.checkpoint))
    allowed_shas = {str(x.get("checkpoint_sha256")) for x in freeze_doc.get("frozen_checkpoints", [])}
    if ckpt_sha not in allowed_shas:
        raise RuntimeError(
            f"Checkpoint SHA-256 {ckpt_sha} is NOT listed in frozen_checkpoints of {freeze_path.name}!"
        )
    return _run_bucket_evaluation(
        args,
        contract,
        "test_spatial_unexposed",
        unseal_freeze_manifest=freeze_doc,
    )


@torch.no_grad()
def cmd_benchmark_latency(args: argparse.Namespace) -> Dict[str, Any]:
    """Matched-domain latency benchmark (`P0-7` / `G6`): neural arms vs exact `4 km` `ExactSceneBVH`."""
    if int(getattr(args, "timed_iters", 0)) <= 0:
        raise ValueError(
            f"cmd_benchmark_latency requires timed_iters >= 1, got {getattr(args, 'timed_iters', None)}"
        )
    skip_phys = bool(getattr(args, "skip_physical_baseline", False))
    if not skip_phys and int(getattr(args, "physical_timed_iters", 0)) <= 0:
        raise ValueError(
            f"cmd_benchmark_latency requires physical_timed_iters >= 1 when physical baseline is enabled, "
            f"got {getattr(args, 'physical_timed_iters', None)}"
        )

    contract = _load_contract_from_args(args)
    dev = torch.device(args.device)
    guard = HostResourceGuard(device=dev)
    guard.check("cmd_benchmark_latency:init")

    model, prov = load_rind_checkpoint(
        args.checkpoint,
        device=dev,
        expected_split_sha256=contract.split_sha256,
        expected_manifest_sha256=contract.manifest_sha256,
    )
    model.eval()
    head_norms = compute_head_parameter_norms(model)
    if head_norms.get("bce_head", 0.0) <= 0.0:
        raise RuntimeError("Latency benchmark requires trained checkpoint with non-zero bce_head!")
    if (
        not bool(prov.get("calibrated", False))
        and not bool(getattr(args, "diagnostic_mode", False))
        and not bool(getattr(args, "allow_uncalibrated", False))
    ):
        raise RuntimeError(
            "Formal latency benchmark requires a calibrated checkpoint (`calibrated=True`)! "
            "Run `calibrate` first, or pass `--diagnostic-mode` / `--allow-uncalibrated`."
        )

    reader = FullDayTileReader(contract, global_mesh_shared_dir=args.global_mesh_dir)
    week_indices = resolve_week_indices(args.temporal_protocol)[: int(args.num_days)]

    bucket_name = str(getattr(args, "bucket", "val_spatial_clean"))
    eval_tiles = contract.get_eval_tiles(bucket_name, model_provenance=prov)
    target_tile = args.tile_id if getattr(args, "tile_id", None) else eval_tiles[0]
    contract.assert_tile_in_bucket(target_tile, bucket_name)

    m_cfg = model.get_model_config()
    crop_sz = int(m_cfg.get("crop_size", 64))
    convgru_mode = str(
        m_cfg.get(
            "spatial_execution_mode",
            "full_tile_512_halo" if model.model_type == "full_tile_convgru" else "per_receiver_crop",
        )
    )
    need_cands = model.model_type.startswith(("rind_","canonical37_"))
    need_crops = model.model_type in ("local_convgru_grid", "full_tile_convgru")

    # 0. Pre-freeze matched evaluation domain (frames, timestamps, receivers_xyz, sun_dirs, valid_mask, 4km scene)
    domain_traj = reader.load_tile_fullday(
        target_tile,
        week_indices=week_indices,
        receiver_mode=args.receiver_mode,
        max_receivers=args.max_receivers,
        max_candidates=args.max_candidates,
        num_azimuth_bins=args.num_azimuth_bins,
        bbox_margin_m=args.bbox_margin_m,
    )
    num_frames = int(domain_traj.sun_dirs.shape[0])
    num_receivers = int(domain_traj.receivers_xyz.shape[0])
    valid_pf = int(domain_traj.valid_mask.sum())
    if valid_pf <= 0:
        raise RuntimeError(f"Zero valid point-frames on {target_tile} for latency benchmark!")

    def _compute_domain_sha(
        tid: str,
        ts_arr: np.ndarray,
        rc_arr: np.ndarray,
        rec_xyz_arr: np.ndarray,
        suns_arr: np.ndarray,
        vm_arr: np.ndarray,
    ) -> str:
        h = hashlib.sha256()
        h.update(tid.encode("utf-8"))
        h.update(np.ascontiguousarray(ts_arr, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(rc_arr, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(rec_xyz_arr, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(suns_arr, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(vm_arr, dtype=bool).tobytes())
        return h.hexdigest()

    frozen_valid_mask = domain_traj.valid_mask.copy()
    frozen_domain_sha = _compute_domain_sha(
        target_tile,
        domain_traj.timestamps_unix_s,
        domain_traj.receiver_rc,
        domain_traj.receivers_xyz,
        domain_traj.sun_dirs,
        frozen_valid_mask,
    )

    # 1. Arm A: `neural_cold_e2e` — clear cache, load ONLY static + condition + 4km DS3MESH (`include_labels=False`, 0 label reads!)
    reader.cache.clear()
    cold_label_reads = 0
    orig_decode_states = reader._decode_record_states

    def _forbid_decode_states(*a: Any, **kw: Any) -> Any:
        nonlocal cold_label_reads
        cold_label_reads += 1
        return orig_decode_states(*a, **kw)

    reader._decode_record_states = _forbid_decode_states  # type: ignore[method-assign]
    try:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_cold0 = time.perf_counter()
        cold_traj = reader.load_tile_fullday(
            target_tile,
            week_indices=week_indices,
            receiver_mode=args.receiver_mode,
            max_receivers=args.max_receivers,
            max_candidates=args.max_candidates,
            num_azimuth_bins=args.num_azimuth_bins,
            bbox_margin_m=args.bbox_margin_m,
            include_labels=False,
        )
        cold_raw: Dict[str, Any] = {
            "tile_ids": [cold_traj.tile_id],
            "sun_dirs": torch.from_numpy(cold_traj.sun_dirs),
            "azimuth_bin_idx": torch.from_numpy(cold_traj.azimuth_bin_idx.astype(np.int64)),
            "same_day_adjacent_mask": torch.from_numpy(cold_traj.same_day_adjacent_mask),
            "local_features": torch.from_numpy(cold_traj.local_features),
            "receivers_xyz": torch.from_numpy(cold_traj.receivers_xyz),
            "receiver_obs_mask": torch.from_numpy(cold_traj.receiver_obs_mask),
            "receiver_rc": torch.from_numpy(cold_traj.receiver_rc),
        }
        if need_cands:
            cold_raw["az_rel_triangles"] = torch.from_numpy(cold_traj.az_rel_triangles)
            cold_raw["az_opaque"] = torch.from_numpy(cold_traj.az_opaque)
            cold_raw["az_mask"] = torch.from_numpy(cold_traj.az_mask)
            cold_raw["az_source_ids"] = torch.from_numpy(cold_traj.az_source_ids.view(np.int64))
            if cold_traj.ray_candidates:
                from .ray_candidates import RayCandidateScene
                scene = RayCandidateScene(cold_traj.scene_triangles_xyz,cold_traj.scene_opaque,cold_traj.scene_source_ids)
                values = scene.query(cold_traj.receivers_xyz,cold_traj.sun_dirs,cold_traj.bbox_margin_m,
                                     args.max_candidates,cold_traj.receiver_source_triangle_ids)
                for key, value in zip(("az_rel_triangles","az_opaque","az_mask","az_source_ids"),values):
                    cold_raw[key] = torch.from_numpy(value)
                cold_raw["azimuth_bin_idx"] = torch.arange(len(cold_traj.sun_dirs))
            from .models import compute_tile_global_knn_pack_torch
            knn = compute_tile_global_knn_pack_torch(cold_raw["receivers_xyz"],cold_raw["receiver_obs_mask"],
                                                     cold_raw["local_features"])
            cold_raw.update({"global_"+key:value for key,value in knn.items()})
        if need_crops:
            cold_raw["static_fields_8hw"] = torch.from_numpy(cold_traj.static_fields_8hw)
            cold_raw["geometry_known_hw"] = torch.from_numpy(cold_traj.geometry_known_hw)
            if convgru_mode == "per_receiver_crop":
                cold_raw["receiver_crops"] = torch.from_numpy(cold_traj.extract_centered_crops(crop_size=crop_sz))
        cold_dev = materialize_gpu_tokens_from_azimuth_bank(
            move_batch_to_device(cold_raw, dev, pin_and_async=(dev.type == "cuda"))
        )
        cold_out = model(LabelReadForbiddenBatchView(cold_dev), use_calibrated_temperature=True)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        cold_sec = max(time.perf_counter() - t_cold0, 1e-9)
    finally:
        reader._decode_record_states = orig_decode_states  # type: ignore[method-assign]

    if cold_label_reads != 0:
        raise RuntimeError(f"Cold E2E latency arm illegally read {cold_label_reads} ground-truth label chunks!")
    if int(cold_out["nonfinite_count"]) != 0:
        raise RuntimeError("Non-finite output during cold latency benchmark!")
    cold_domain_sha = _compute_domain_sha(
        cold_traj.tile_id,
        cold_traj.timestamps_unix_s,
        cold_traj.receiver_rc,
        cold_traj.receivers_xyz,
        cold_traj.sun_dirs,
        frozen_valid_mask,
    )
    if cold_domain_sha != frozen_domain_sha:
        raise RuntimeError("Cold E2E domain SHA mismatch against frozen domain manifest!")

    # 2. Prepare verified shard for Arm B (`neural_cached_shard_e2e`) & Arm C (`neural_pure_warm_gpu_forward`)
    shard = build_or_open_single_tile_shard(
        args.cache_dir,
        reader,
        target_tile,
        week_indices=week_indices,
        receiver_mode=args.receiver_mode,
        max_receivers=args.max_receivers,
        max_candidates=args.max_candidates,
        num_azimuth_bins=args.num_azimuth_bins,
        bbox_margin_m=args.bbox_margin_m,
    )
    shard_domain_sha = _compute_domain_sha(
        shard.tile_id,
        shard._mmap_slice_copy("timestamps_unix_s"),
        shard._mmap_slice_copy("receiver_rc"),
        shard._mmap_slice_copy("receivers_xyz"),
        shard._mmap_slice_copy("sun_dirs"),
        shard._mmap_slice_copy("valid_mask"),
    )
    if shard_domain_sha != frozen_domain_sha:
        raise RuntimeError("Shard domain SHA mismatch against frozen domain manifest!")

    # Arm B: `neural_cached_shard_e2e` (mmap read without labels + H2D + GPU token build + model forward)
    shard_label_reads = 0
    orig_mmap_slice = shard._mmap_slice_copy

    def _guarded_mmap_slice(name: str, sl: Any = slice(None)) -> np.ndarray:
        nonlocal shard_label_reads
        if name in (
            "truth",
            "valid_mask",
            "boundary_distance_m",
            "boundary_distance_valid",
            "response_5x5_bitpack",
        ):
            shard_label_reads += 1
        return orig_mmap_slice(name, sl)

    shard._mmap_slice_copy = _guarded_mmap_slice  # type: ignore[method-assign]
    cached_exec_counters: Optional[Dict[str, Any]] = None
    try:
        for _ in range(int(args.warmup_iters)):
            sb = shard.load_compact_batch(
                include_candidate_bank=need_cands,
                include_receiver_crops=need_crops,
                include_labels=False,
                crop_size=crop_sz,
                convgru_execution_mode=convgru_mode,
            )
            wb = materialize_gpu_tokens_from_azimuth_bank(
                move_batch_to_device(sb, dev, pin_and_async=(dev.type == "cuda"))
            )
            _ = model(LabelReadForbiddenBatchView(wb), use_calibrated_temperature=True)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

        reset_fast_data_io_stats()
        cached_secs: List[float] = []
        for _ in range(int(args.timed_iters)):
            tc0 = time.perf_counter()
            sb = shard.load_compact_batch(
                include_candidate_bank=need_cands,
                include_receiver_crops=need_crops,
                include_labels=False,
                crop_size=crop_sz,
                convgru_execution_mode=convgru_mode,
            )
            wb = materialize_gpu_tokens_from_azimuth_bank(
                move_batch_to_device(sb, dev, pin_and_async=(dev.type == "cuda"))
            )
            cached_out = model(LabelReadForbiddenBatchView(wb), use_calibrated_temperature=True)
            cached_exec_counters = cached_out.get("execution_counters")
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            cached_secs.append(time.perf_counter() - tc0)
        cached_io_stats = get_fast_data_io_stats()
    finally:
        shard._mmap_slice_copy = orig_mmap_slice  # type: ignore[method-assign]

    if shard_label_reads != 0:
        raise RuntimeError(f"Cached shard inference arm illegally read {shard_label_reads} label arrays!")
    cached_mean_sec = float(np.mean(cached_secs)) if cached_secs else 1e-9

    # 3. Arm C: Counter-instrumented `neural_pure_warm_gpu_forward` (0 file opens, 0 mmap reads, 0 H2D, 0 token builds)
    prestaged_cpu = shard.load_compact_batch(
        include_candidate_bank=need_cands,
        include_receiver_crops=need_crops,
        include_labels=False,
        crop_size=crop_sz,
        convgru_execution_mode=convgru_mode,
    )
    prestaged_gpu = LabelReadForbiddenBatchView(
        materialize_gpu_tokens_from_azimuth_bank(
            move_batch_to_device(prestaged_cpu, dev, pin_and_async=(dev.type == "cuda"))
        )
    )
    for _ in range(int(args.warmup_iters)):
        _ = model(prestaged_gpu, use_calibrated_temperature=True)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    warm_counters = {
        "file_opens": 0,
        "mmap_reads": 0,
        "h2d_transfers": 0,
        "token_builds": 0,
        "label_reads": 0,
    }
    orig_open = builtins.open

    def _counted_open(*a: Any, **kw: Any):
        warm_counters["file_opens"] += 1
        return orig_open(*a, **kw)

    def _counted_mmap(name: str, sl: Any = slice(None)) -> np.ndarray:
        warm_counters["mmap_reads"] += 1
        return orig_mmap_slice(name, sl)

    warm_secs: List[float] = []
    warm_exec_counters: Optional[Dict[str, Any]] = None
    reset_fast_data_io_stats()
    builtins.open = _counted_open  # type: ignore[assignment]
    shard._mmap_slice_copy = _counted_mmap  # type: ignore[method-assign]
    try:
        for _ in range(int(args.timed_iters)):
            tw0 = time.perf_counter()
            warm_out = model(prestaged_gpu, use_calibrated_temperature=True)
            warm_exec_counters = warm_out.get("execution_counters")
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            warm_secs.append(time.perf_counter() - tw0)
        warm_io_stats = get_fast_data_io_stats()
    finally:
        builtins.open = orig_open  # type: ignore[assignment]
        shard._mmap_slice_copy = orig_mmap_slice  # type: ignore[method-assign]

    if any(v != 0 for v in warm_counters.values()) or any(v != 0 for v in warm_io_stats.values()):
        raise RuntimeError(
            f"Pure warm GPU forward loop violated zero-IO/H2D contract: counters={warm_counters}, io={warm_io_stats}"
        )
    warm_mean_sec = float(np.mean(warm_secs)) if warm_secs else 1e-9

    # 4. Arm D: Matched-domain `physical_exact_4km_raytrace` via `ExactSceneBVH` (unless explicitly skipped)
    physical_arm: Optional[Dict[str, Any]] = None
    rel_speedups: Optional[Dict[str, float]] = None
    if not skip_phys:
        from .inverse_design import DEFAULT_T_MIN_M, ExactSceneBVH

        tbvh0 = time.perf_counter()
        bvh = ExactSceneBVH(
            domain_traj.scene_triangles_xyz,
            opaque_mask=domain_traj.scene_opaque,
            source_triangle_ids=domain_traj.scene_source_ids,
        )
        bvh_build_sec = time.perf_counter() - tbvh0

        phys_domain_sha = _compute_domain_sha(
            domain_traj.tile_id,
            domain_traj.timestamps_unix_s,
            domain_traj.receiver_rc,
            domain_traj.receivers_xyz,
            domain_traj.sun_dirs,
            frozen_valid_mask,
        )
        if phys_domain_sha != frozen_domain_sha:
            raise RuntimeError("Physical BVH domain SHA mismatch!")

        phys_warmup = min(int(args.warmup_iters), 1)
        phys_iters = int(getattr(args, "physical_timed_iters", 1))
        for _ in range(phys_warmup):
            _ = bvh.query_trajectory_sunlit(
                domain_traj.receivers_xyz,
                domain_traj.sun_dirs,
                t_min=DEFAULT_T_MIN_M,
                exclude_source_triangle_ids=domain_traj.receiver_source_triangle_ids,
            )

        phys_secs: List[float] = []
        for _ in range(phys_iters):
            tp0 = time.perf_counter()
            _ = bvh.query_trajectory_sunlit(
                domain_traj.receivers_xyz,
                domain_traj.sun_dirs,
                t_min=DEFAULT_T_MIN_M,
                exclude_source_triangle_ids=domain_traj.receiver_source_triangle_ids,
            )
            phys_secs.append(time.perf_counter() - tp0)
        phys_mean_sec = max(float(np.mean(phys_secs)), 1e-9)
        physical_arm = {
            "backend": "ExactSceneBVH_float64_MollerTrumbore_CPU",
            "device": "cpu",
            "t_min_m": DEFAULT_T_MIN_M,
            "scene_horizon_m": float(domain_traj.bbox_margin_m),
            "scene_triangle_count": int(domain_traj.scene_triangles_xyz.shape[0]),
            "bvh_build_wall_seconds": bvh_build_sec,
            "warmup_iters": phys_warmup,
            "timed_iters": phys_iters,
            "wall_seconds_list": phys_secs,
            "mean_wall_seconds": phys_mean_sec,
            "ms_per_tile_frame": 1000.0 * phys_mean_sec / max(num_frames, 1),
            "us_per_valid_point_frame": 1e6 * phys_mean_sec / max(valid_pf, 1),
            "domain_sha256": phys_domain_sha,
        }
        rel_speedups = {
            "pure_warm_gpu_vs_physical_exact": phys_mean_sec / warm_mean_sec,
            "cached_shard_e2e_vs_physical_exact": phys_mean_sec / cached_mean_sec,
            "cold_e2e_vs_physical_exact": phys_mean_sec / cold_sec,
        }

    guard.check("cmd_benchmark_latency:done")
    receipt = {
        "benchmark_contract": BENCHMARK_CONTRACT_ID,
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": compute_file_sha256(Path(args.checkpoint)),
        "model_type": model.model_type,
        "ablation_tier": model.ablation_tier,
        "model_config": model.get_model_config(),
        "head_parameter_norms": head_norms,
        "device": str(dev),
        "matched_domain": {
            "tile_id": target_tile,
            "bucket": bucket_name,
            "domain_sha256": frozen_domain_sha,
            "days_evaluated": len(week_indices),
            "frames_evaluated": num_frames,
            "frames_per_day": num_frames,
            "receivers_evaluated": num_receivers,
            "valid_point_frames": valid_pf,
            "scene_horizon_m": float(domain_traj.bbox_margin_m),
            "scene_triangle_count": int(domain_traj.scene_triangles_xyz.shape[0]),
        },
        "label_access_audit": {
            "untimed_domain_setup_reads_valid_mask_for_denominator": True,
            "timed_neural_cold_e2e_label_reads": cold_label_reads,
            "timed_neural_cached_shard_e2e_label_reads": shard_label_reads,
            "timed_neural_pure_warm_gpu_forward_label_reads": int(warm_counters["label_reads"]),
            "forward_wrapped_in_label_read_forbidden_view": True,
        },
        "neural_cold_e2e": {
            "clock_boundary": "uncached_static_condition_4km_mesh_read_plus_h2d_plus_token_build_plus_forward",
            "label_chunk_reads": cold_label_reads,
            "wall_seconds": cold_sec,
            "ms_per_tile_frame": 1000.0 * cold_sec / max(num_frames, 1),
            "us_per_valid_point_frame": 1e6 * cold_sec / max(valid_pf, 1),
            "domain_sha256": cold_domain_sha,
            "execution_counters": cold_out.get("execution_counters"),
        },
        "neural_cached_shard_e2e": {
            "clock_boundary": "mmap_input_read_plus_h2d_plus_gpu_token_build_plus_forward",
            "label_array_reads": shard_label_reads,
            "fast_data_io_stats": cached_io_stats,
            "execution_counters": cached_exec_counters,
            "warmup_iters": int(args.warmup_iters),
            "timed_iters": int(args.timed_iters),
            "wall_seconds_list": cached_secs,
            "mean_wall_seconds": cached_mean_sec,
            "ms_per_tile_frame": 1000.0 * cached_mean_sec / max(num_frames, 1),
            "us_per_valid_point_frame": 1e6 * cached_mean_sec / max(valid_pf, 1),
            "domain_sha256": shard_domain_sha,
        },
        "neural_pure_warm_gpu_forward": {
            "clock_boundary": "gpu_resident_model_forward_only_cuda_synchronized",
            "warm_loop_io_counters": warm_counters,
            "fast_data_io_stats": warm_io_stats,
            "execution_counters": warm_exec_counters,
            "warmup_iters": int(args.warmup_iters),
            "timed_iters": int(args.timed_iters),
            "wall_seconds_list": warm_secs,
            "mean_wall_seconds": warm_mean_sec,
            "ms_per_tile_frame": 1000.0 * warm_mean_sec / max(num_frames, 1),
            "us_per_valid_point_frame": 1e6 * warm_mean_sec / max(valid_pf, 1),
            "domain_sha256": shard_domain_sha,
        },
        "physical_exact_4km_raytrace": physical_arm,
        "relative_speedup_vs_physical": rel_speedups,
        # Backwards-compatible aliases
        "cold_start": {
            "wall_seconds": cold_sec,
            "ms_per_tile_frame": 1000.0 * cold_sec / max(num_frames, 1),
            "us_per_valid_point_frame": 1e6 * cold_sec / max(valid_pf, 1),
        },
        "warm_rollout": {
            "timed_iters": int(args.timed_iters),
            "mean_wall_seconds": cached_mean_sec,
            "ms_per_tile_frame": 1000.0 * cached_mean_sec / max(num_frames, 1),
            "us_per_valid_point_frame": 1e6 * cached_mean_sec / max(valid_pf, 1),
        },
        "host_rss_mib": get_process_rss_mib(),
        "resource_telemetry": guard.summary(),
        "completed_at_utc": _utc_now_iso(),
    }
    if args.output:
        _write_json_atomic(args.output, receipt)
    if not args.quiet:
        print(json.dumps(receipt, indent=2))
    return receipt


def cmd_benchmark_train_speed(args: argparse.Namespace) -> Dict[str, Any]:
    if int(getattr(args, "timed_iters", 0)) <= 0:
        raise ValueError(
            f"cmd_benchmark_train_speed requires timed_iters >= 1, got {getattr(args, 'timed_iters', None)}"
        )
    contract = _load_contract_from_args(args)
    dev = torch.device(args.device)
    guard = HostResourceGuard(device=dev)
    guard.check("cmd_benchmark_train_speed:init")

    fit_tiles = contract.get_fit_tiles("fit_8")
    target_tile = args.tile_id if getattr(args, "tile_id", None) else fit_tiles[0]
    contract.assert_fit_allowed([target_tile])
    firewall_token = contract.issue_fit_firewall_token([target_tile], subset_name="fit_8")

    reader = FullDayTileReader(contract, global_mesh_shared_dir=args.global_mesh_dir)
    week_indices = resolve_week_indices(args.temporal_protocol)[: int(args.num_days)]

    t_prep = time.perf_counter()
    shard = build_or_open_single_tile_shard(
        args.cache_dir,
        reader,
        target_tile,
        week_indices=week_indices,
        receiver_mode=args.receiver_mode,
        max_receivers=args.max_receivers,
        max_candidates=args.max_candidates,
        num_azimuth_bins=args.num_azimuth_bins,
        bbox_margin_m=args.bbox_margin_m,
    )
    prep_sec = time.perf_counter() - t_prep
    crop_sz = int(getattr(args, "crop_size", 64))
    need_cands = args.model_type.startswith(("rind_","canonical37_"))
    need_crops = args.model_type in ("local_convgru_grid", "full_tile_convgru")
    convgru_exec_mode = "full_tile_512_halo" if args.model_type == "full_tile_convgru" else "per_receiver_crop"

    def _baseline_loader() -> Dict[str, Any]:
        reader.cache.clear()
        traj = reader.load_tile_fullday(
            target_tile,
            week_indices=week_indices,
            receiver_mode=args.receiver_mode,
            max_receivers=args.max_receivers,
            max_candidates=args.max_candidates,
            num_azimuth_bins=args.num_azimuth_bins,
            bbox_margin_m=args.bbox_margin_m,
        )
        return traj.materialize_canonical37_batch(crop_size=crop_sz)

    def _accelerated_loader() -> Dict[str, Any]:
        return shard.load_compact_batch(
            include_candidate_bank=need_cands,
            include_receiver_crops=need_crops,
            crop_size=crop_sz,
            convgru_execution_mode=convgru_exec_mode,
        )

    def _factory():
        torch.manual_seed(20261111)
        attention_options = _candidate_attention_options(args) if args.model_type.startswith("rind_") else {}
        return build_model(
            args.model_type,
            hidden_dim=int(args.hidden_dim),
            max_candidates=int(args.max_candidates),
            crop_size=crop_sz,
            meters_per_pixel=float(shard.meters_per_pixel),
            **attention_options,
        )

    perf = benchmark_training_speed(
        _factory,
        _baseline_loader,
        _accelerated_loader,
        firewall_token=firewall_token,
        expected_split_sha256=contract.split_sha256,
        prepare_gpu_tokens_fn=materialize_gpu_tokens_from_azimuth_bank,
        device=dev,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
        enable_compile=args.enable_compile,
        shard_build_wall_seconds=prep_sec,
        resource_guard=guard,
        cf_subgroup_size=int(getattr(args, "cf_subgroup_size", 0)),
    )
    perf["benchmark_contract"] = BENCHMARK_CONTRACT_ID
    perf["model_type"] = args.model_type
    perf["temporal_protocol"] = args.temporal_protocol
    perf["tile_id"] = target_tile
    perf["days_per_step"] = len(week_indices)
    perf["frames_per_step"] = int(shard.num_frames)
    perf["receivers_per_step"] = int(shard.num_receivers)
    perf["one_time_mmap_build_seconds"] = prep_sec
    perf["completed_at_utc"] = _utc_now_iso()

    if args.output:
        _write_json_atomic(args.output, perf)
    if not args.quiet:
        print(json.dumps(perf, indent=2))
    return perf


def _inverse_bucket_tiles(contract, bucket):
    if bucket in CANONICAL_EVAL_BUCKETS:
        return contract.get_eval_tiles(bucket)
    if bucket in FIT_SUBSET_SIZES:
        return contract.get_fit_tiles(bucket)
    if bucket == "calibration":
        return contract.get_calibration_tiles()
    raise ValueError(f"Unsupported inverse bucket {bucket!r}")


def cmd_inverse_design(args: argparse.Namespace) -> Dict[str, Any]:
    from .inverse_design import aggregate_multi_tile_inverse_design, optimize_fullday_building_heights

    contract = _load_contract_from_args(args)
    guard = HostResourceGuard(device=args.device)
    guard.check("cmd_inverse_design:init")

    bucket_name = str(getattr(args, "bucket", "val_spatial_clean"))
    if bucket_name == "test_spatial_unexposed":
        raise RuntimeError(
            "BLIND TEST LOCK: `inverse-design` on `test_spatial_unexposed` is prohibited during development!"
        )
    bucket_tiles = _inverse_bucket_tiles(contract,bucket_name)

    raw_tile_ids = getattr(args, "tile_ids", None)
    if raw_tile_ids:
        target_tiles = [t.strip() for t in str(raw_tile_ids).split(",") if t.strip()]
    elif getattr(args, "tile_id", None):
        target_tiles = [str(args.tile_id)]
    else:
        target_tiles = [bucket_tiles[0]]

    for tid in target_tiles:
        if tid not in bucket_tiles:
            raise ValueError(f"Tile {tid!r} is outside requested bucket {bucket_name!r}")

    diag_mode = bool(getattr(args, "diagnostic_mode", False))
    allow_uncal = bool(getattr(args, "allow_uncalibrated", False))
    rind_mod = None
    ckpt_sha = None
    if getattr(args, "checkpoint", None) is not None:
        ckpt_sha = compute_file_sha256(Path(args.checkpoint))
        rind_mod, ckpt_prov = load_rind_checkpoint(
            args.checkpoint,
            device=args.device,
            expected_split_sha256=contract.split_sha256,
            expected_manifest_sha256=contract.manifest_sha256,
        )
        if not bool(ckpt_prov.get("calibrated", False)) and not diag_mode and not allow_uncal:
            raise RuntimeError(
                "Formal inverse-design with `--checkpoint` requires a calibrated checkpoint (`calibrated=True`)! "
                "Run `calibrate` first, or pass `--diagnostic-mode` / `--allow-uncalibrated`."
            )
        if bucket_name in CANONICAL_EVAL_BUCKETS:
            contract.assert_eval_bucket_allowed(bucket_name, target_tiles, model_provenance=ckpt_prov)

    reader = FullDayTileReader(contract, global_mesh_shared_dir=args.global_mesh_dir)
    week_indices = resolve_week_indices(args.temporal_protocol)[: int(getattr(args, "num_days", 1) or 1)]
    enforce_vol = not bool(getattr(args, "disable_volume_conservation", False))
    allow_nonclosed = diag_mode or bool(getattr(args, "allow_nonclosed_surface_proxy", False))
    max_oracle_q = getattr(args, "max_oracle_queries", None)
    max_wall_s = getattr(args, "max_wall_clock_s", None)

    per_tile_results: List[Any] = []
    first_traj = None
    for tid in target_tiles:
        traj = reader.load_tile_fullday(
            tid,
            week_indices=week_indices,
            receiver_mode=args.receiver_mode,
            max_receivers=args.max_receivers,
            max_candidates=args.max_candidates,
            bbox_margin_m=args.bbox_margin_m,
        )
        if first_traj is None:
            first_traj = traj
        c_sz = getattr(rind_mod, "crop_size", None)
        rec_crops = (
            traj.extract_centered_crops(crop_size=c_sz)
            if c_sz is not None and len(traj.receivers_xyz)
            else None
        )
        inv_res_i = optimize_fullday_building_heights(
            traj.receivers_xyz,
            traj.sun_dirs,
            traj.scene_triangles_xyz,
            reference_trajectory=traj,
            source_instance_ids=traj.scene_source_instance_ids,
            ground_mask=traj.scene_ground_mask,
            opaque_mask=traj.scene_opaque,
            source_triangle_ids=traj.scene_source_ids,
            exclude_source_triangle_ids=traj.receiver_source_triangle_ids,
            solar_regimes=traj.solar_regimes,
            valid_mask=traj.valid_mask,
            same_day_adjacent_mask=traj.same_day_adjacent_mask,
            local_features=traj.local_features,
            receiver_crops=rec_crops,
            grid_step_xz=(float(np.median(np.diff(traj.world_x_m))), float(np.median(np.diff(traj.world_z_m))))
            if traj.world_x_m is not None and len(traj.world_x_m)>1 and traj.world_z_m is not None and len(traj.world_z_m)>1 else None,
            grid_bounds_xz=(float(traj.world_x_m.min()), float(traj.world_x_m.max()),
                            float(traj.world_z_m.min()), float(traj.world_z_m.max()))
            if traj.world_x_m is not None and traj.world_z_m is not None else None,
            meters_per_pixel=float(traj.meters_per_pixel),
            optimizer_arm=args.optimizer_arm,
            rind_model=rind_mod,
            checkpoint_sha256=ckpt_sha,
            max_candidates=args.max_candidates,
            editable_radius_m=args.editable_radius_m,
            max_delta_h_m=args.max_delta_h_m,
            steps=args.steps,
            lr=args.lr,
            enforce_volume_conservation=enforce_vol,
            enforce_zero_sum_height=args.zero_sum_height,
            scene_bbox_margin_m=float(traj.bbox_margin_m),
            require_full_4km_horizon=(not diag_mode),
            allow_nonclosed_surface_proxy=allow_nonclosed,
            max_oracle_queries=max_oracle_q,
            max_wall_clock_s=max_wall_s,
            device=args.device,
        )
        per_tile_results.append(inv_res_i)

    assert first_traj is not None
    guard.check("cmd_inverse_design:done")
    receipt: Dict[str, Any] = {
        "benchmark_contract": BENCHMARK_CONTRACT_ID,
        "tile_id": target_tiles[0],
        "tile_ids": target_tiles,
        "bucket": bucket_name,
        "window_ids": list(first_traj.window_ids),
        "scene_horizon_m": float(first_traj.bbox_margin_m),
        "full_scene_triangle_count": int(first_traj.scene_triangles_xyz.shape[0]),
        "inverse_design_result": per_tile_results[0].to_dict(),
        "per_tile_results": [r.to_dict() for r in per_tile_results],
        "multi_tile_summary": aggregate_multi_tile_inverse_design(per_tile_results),
        "resource_telemetry": guard.summary(),
        "completed_at_utc": _utc_now_iso(),
    }
    if args.output:
        _write_json_atomic(args.output, receipt)
    if not args.quiet:
        print(json.dumps(receipt, indent=2))
    return receipt


def cmd_infer_building_height(args: argparse.Namespace) -> Dict[str, Any]:
    """Bayesian inverse unknown-building height estimation under noisy partial shadow observations (`P1`)."""
    from .inverse_design import build_layered_occlusion_benchmark_scene, infer_unknown_building_heights_bayesian

    guard = HostResourceGuard(device=args.device)
    guard.check("cmd_infer_building_height:init")

    rind_mod = None
    ckpt_sha = None
    if bool(getattr(args, "synthetic_benchmark", False)):
        bench = build_layered_occlusion_benchmark_scene()
        if getattr(args, "checkpoint", None) is not None:
            ckpt_sha = compute_file_sha256(Path(args.checkpoint))
            rind_mod, _ = load_rind_checkpoint(args.checkpoint, device=args.device)
        res = infer_unknown_building_heights_bayesian(
            bench["receivers_xyz"],
            bench["sun_directions_xyz"],
            bench["triangles_xyz"],
            source_instance_ids=bench["source_instance_ids"],
            ground_mask=bench["ground_mask"],
            rind_model=rind_mod,
            max_unknown_buildings=int(args.max_unknown_buildings),
            unknown_building_ids=getattr(args,"unknown_building_ids",None),
            heldout_frame_indices=getattr(args,"heldout_frame_indices",()),
            receiver_elevation_public=getattr(args,"receiver_elevation_public",False),
            num_grid_hypotheses=int(args.num_grid_hypotheses),
            height_grid_min_m=getattr(args, "height_grid_min_m", 2.0),
            height_grid_max_m=getattr(args, "height_grid_max_m", 50.0),
            sensor_noise_flip_prob=float(args.sensor_noise),
            initial_queries=int(args.initial_queries),
            active_rounds=int(args.active_rounds),
            queries_per_round=int(args.queries_per_round),
            seed=int(args.seed),
            device=str(args.device),
        )
        tile_tag = "synthetic_layered_plaza_v1"
        bucket_name = "synthetic_benchmark"
    else:
        contract = _load_contract_from_args(args)
        bucket_name = str(getattr(args, "bucket", "val_spatial_clean"))
        if bucket_name == "test_spatial_unexposed":
            raise RuntimeError("BLIND TEST LOCK: `infer-building-height` on `test_spatial_unexposed` is prohibited!")
        bucket_tiles = _inverse_bucket_tiles(contract,bucket_name)
        tile_tag = args.tile_id if getattr(args, "tile_id", None) else bucket_tiles[0]
        if tile_tag not in bucket_tiles:
            raise ValueError(f"Tile {tile_tag!r} is outside requested bucket {bucket_name!r}")
        if getattr(args, "checkpoint", None) is not None:
            ckpt_sha = compute_file_sha256(Path(args.checkpoint))
            rind_mod, ckpt_prov = load_rind_checkpoint(
                args.checkpoint,
                device=args.device,
                expected_split_sha256=contract.split_sha256,
                expected_manifest_sha256=contract.manifest_sha256,
            )
            if bucket_name in CANONICAL_EVAL_BUCKETS:
                contract.assert_eval_bucket_allowed(bucket_name, [tile_tag], model_provenance=ckpt_prov)
        reader = FullDayTileReader(contract, global_mesh_shared_dir=args.global_mesh_dir)
        week_indices = resolve_week_indices(args.temporal_protocol)[: int(getattr(args, "num_days", 1) or 1)]
        az_bins_arg = getattr(args, "num_azimuth_bins", None)
        eff_az_bins = (
            int(az_bins_arg)
            if az_bins_arg is not None
            else (int(getattr(rind_mod, "num_azimuth_bins", 0)) if rind_mod is not None else 0)
        )
        traj = reader.load_tile_fullday(
            tile_tag,
            week_indices=week_indices,
            receiver_mode=args.receiver_mode,
            max_receivers=args.max_receivers,
            max_candidates=args.max_candidates,
            num_azimuth_bins=eff_az_bins,
            bbox_margin_m=args.bbox_margin_m,
        )
        res = infer_unknown_building_heights_bayesian(
            traj.receivers_xyz,
            traj.sun_dirs,
            traj.scene_triangles_xyz,
            reference_trajectory=traj,
            source_instance_ids=traj.scene_source_instance_ids,
            ground_mask=traj.scene_ground_mask,
            opaque_mask=traj.scene_opaque,
            source_triangle_ids=traj.scene_source_ids,
            exclude_source_triangle_ids=traj.receiver_source_triangle_ids,
            valid_mask=traj.valid_mask,
            same_day_adjacent_mask=traj.same_day_adjacent_mask,
            rind_model=rind_mod,
            max_unknown_buildings=int(args.max_unknown_buildings),
            unknown_building_ids=getattr(args,"unknown_building_ids",None),
            heldout_frame_indices=getattr(args,"heldout_frame_indices",()),
            receiver_elevation_public=getattr(args,"receiver_elevation_public",False),
            num_grid_hypotheses=int(args.num_grid_hypotheses),
            height_grid_min_m=getattr(args, "height_grid_min_m", 2.0),
            height_grid_max_m=getattr(args, "height_grid_max_m", 50.0),
            sensor_noise_flip_prob=float(args.sensor_noise),
            initial_queries=int(args.initial_queries),
            active_rounds=int(args.active_rounds),
            queries_per_round=int(args.queries_per_round),
            seed=int(args.seed),
            device=str(args.device),
        )

    guard.check("cmd_infer_building_height:done")
    receipt = {
        "benchmark_contract": BENCHMARK_CONTRACT_ID,
        "tile_id": tile_tag,
        "bucket": bucket_name,
        "checkpoint_sha256": ckpt_sha,
        "height_inference_result": res,
        "resource_telemetry": guard.summary(),
        "completed_at_utc": _utc_now_iso(),
    }
    if args.output:
        _write_json_atomic(args.output, receipt)
    if not args.quiet:
        print(json.dumps(receipt, indent=2))
    return receipt


def cmd_eval_repr_sufficiency(args: argparse.Namespace) -> Dict[str, Any]:
    """Evaluate Representation Sufficiency under Driver Changes & Nonadditive Unseen Edit Combinations."""
    from .inverse_design import build_layered_occlusion_benchmark_scene, evaluate_representation_sufficiency

    guard = HostResourceGuard(device=args.device)
    guard.check("cmd_eval_repr_sufficiency:init")

    rind_mod = None
    ckpt_sha = None
    if bool(getattr(args, "synthetic_benchmark", False)):
        bench = build_layered_occlusion_benchmark_scene()
        if getattr(args, "checkpoint", None) is not None:
            ckpt_sha = compute_file_sha256(Path(args.checkpoint))
            rind_mod, _ = load_rind_checkpoint(args.checkpoint, device=args.device)
        res = evaluate_representation_sufficiency(
            bench["receivers_xyz"],
            bench["sun_directions_xyz"],
            bench["triangles_xyz"],
            rind_model=rind_mod,
            source_instance_ids=bench["source_instance_ids"],
            ground_mask=bench["ground_mask"],
            edit_drop_fraction=float(args.edit_drop_fraction),
            max_building_pairs=int(args.max_building_pairs),
            device=str(args.device),
        )
        tile_tag = "synthetic_layered_plaza_v1"
        bucket_name = "synthetic_benchmark"
    else:
        contract = _load_contract_from_args(args)
        bucket_name = str(getattr(args, "bucket", "val_spatial_clean"))
        if bucket_name == "test_spatial_unexposed":
            raise RuntimeError("BLIND TEST LOCK: `eval-repr-sufficiency` on `test_spatial_unexposed` is prohibited!")
        bucket_tiles = _inverse_bucket_tiles(contract,bucket_name)
        tile_tag = args.tile_id if getattr(args, "tile_id", None) else bucket_tiles[0]
        if tile_tag not in bucket_tiles:
            raise ValueError(f"Tile {tile_tag!r} is outside requested bucket {bucket_name!r}")
        if getattr(args, "checkpoint", None) is not None:
            ckpt_sha = compute_file_sha256(Path(args.checkpoint))
            rind_mod, ckpt_prov = load_rind_checkpoint(
                args.checkpoint,
                device=args.device,
                expected_split_sha256=contract.split_sha256,
                expected_manifest_sha256=contract.manifest_sha256,
            )
            if bucket_name in CANONICAL_EVAL_BUCKETS:
                contract.assert_eval_bucket_allowed(bucket_name, [tile_tag], model_provenance=ckpt_prov)
        reader = FullDayTileReader(contract, global_mesh_shared_dir=args.global_mesh_dir)
        week_indices = resolve_week_indices(args.temporal_protocol)[: int(getattr(args, "num_days", 1) or 1)]
        az_bins_arg = getattr(args, "num_azimuth_bins", None)
        eff_az_bins = (
            int(az_bins_arg)
            if az_bins_arg is not None
            else (int(getattr(rind_mod, "num_azimuth_bins", 0)) if rind_mod is not None else 0)
        )
        traj = reader.load_tile_fullday(
            tile_tag,
            week_indices=week_indices,
            receiver_mode=args.receiver_mode,
            max_receivers=args.max_receivers,
            max_candidates=args.max_candidates,
            num_azimuth_bins=eff_az_bins,
            bbox_margin_m=args.bbox_margin_m,
        )
        res = evaluate_representation_sufficiency(
            traj.receivers_xyz,
            traj.sun_dirs,
            traj.scene_triangles_xyz,
            reference_trajectory=traj,
            rind_model=rind_mod,
            source_instance_ids=traj.scene_source_instance_ids,
            ground_mask=traj.scene_ground_mask,
            opaque_mask=traj.scene_opaque,
            source_triangle_ids=traj.scene_source_ids,
            exclude_source_triangle_ids=traj.receiver_source_triangle_ids,
            valid_mask=traj.valid_mask,
            same_day_adjacent_mask=traj.same_day_adjacent_mask,
            edit_drop_fraction=float(args.edit_drop_fraction),
            max_building_pairs=int(args.max_building_pairs),
            device=str(args.device),
        )

    guard.check("cmd_eval_repr_sufficiency:done")
    receipt = {
        "benchmark_contract": BENCHMARK_CONTRACT_ID,
        "tile_id": tile_tag,
        "bucket": bucket_name,
        "checkpoint_sha256": ckpt_sha,
        "representation_sufficiency_result": res,
        "resource_telemetry": guard.summary(),
        "completed_at_utc": _utc_now_iso(),
    }
    if args.output:
        _write_json_atomic(args.output, receipt)
    if not args.quiet:
        print(json.dumps(receipt, indent=2))
    return receipt


def cmd_run_dev_suite(args: argparse.Namespace) -> Dict[str, Any]:
    """Run strictly development-only (`fit` -> `calibration` -> `val_spatial_clean`) multi-model suite (`G5`)."""
    contract = _load_contract_from_args(args)
    receipt_class = _enforce_publication_or_diagnostic_mode(args, contract)

    out_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "mmap_cache"
    ledger_path = out_dir / "val_exposure_ledger.json"

    fit_tiles = contract.get_fit_tiles(args.fit_subset)
    cal_tiles = contract.get_calibration_tiles()
    val_tiles = contract.get_eval_tiles("val_spatial_clean", allow_empty=True)
    if args.max_tiles > 0:
        fit_tiles = fit_tiles[: int(args.max_tiles)]
        cal_tiles = cal_tiles[: int(args.max_tiles)]
        val_tiles = val_tiles[: int(args.max_tiles)]

    week_indices = resolve_week_indices(args.temporal_protocol)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    selection_rule = str(getattr(args, "selection_rule", "fixed_final_epoch"))

    models_to_run = [
        ("rind_full_geom", "rind_full_geom", 64),
        ("rind_no_edge_margin", "rind_no_edge_margin", 64),
        ("rind_zero_margin", "rind_zero_margin", 64),
        ("local_convgru_grid", "local_convgru_grid", 64),
        ("pointwise_mlp", "pointwise_mlp", 64),
    ]

    est_days_per_step = max(1, int(getattr(args, "days_per_batch", 1) or 1))
    est_steps_per_epoch = len(fit_tiles) * max(1, math.ceil(len(week_indices) / est_days_per_step))
    execution_plan = {
        "benchmark_contract": BENCHMARK_CONTRACT_ID,
        "receipt_class": receipt_class,
        "split_counts": {
            "fit": len(fit_tiles),
            "calibration": len(cal_tiles),
            "val_spatial_clean": len(val_tiles),
            "test_spatial_unexposed_blind_locked": contract.build_verification_report()[
                "counts_by_canonical_split"
            ]["test_spatial_unexposed"],
        },
        "temporal_protocol": args.temporal_protocol,
        "weeks_per_tile": len(week_indices),
        "receiver_mode": args.receiver_mode,
        "max_receivers": int(args.max_receivers),
        "model_configurations": [m[0] for m in models_to_run],
        "seeds": seeds,
        "selection_rule": selection_rule,
        "epochs_per_run": int(args.epochs),
        "estimated_updates_per_run": int(est_steps_per_epoch * int(args.epochs)),
    }
    _write_json_atomic(out_dir / "00_pre_gpu_execution_plan.json", execution_plan)
    if not args.quiet:
        print(json.dumps({"PRE_GPU_EXECUTION_PLAN": execution_plan}, indent=2))

    split_report = cmd_verify_split(
        argparse.Namespace(
            split_file=args.split_file,
            manifest_file=args.manifest_file,
            allow_synthetic_split=args.allow_synthetic_split,
            diagnostic_mode=args.diagnostic_mode,
            global_mesh_dir=args.global_mesh_dir,
            output=out_dir / "01_split_verification.json",
            quiet=True,
        )
    )

    model_runs: List[Dict[str, Any]] = []
    by_model_seed_summaries: Dict[str, List[Dict[str, Any]]] = {m[0]: [] for m in models_to_run}
    by_seed_model_tile_accs: Dict[int, Dict[str, Dict[str, ConfusionAccumulator]]] = {s: {} for s in seeds}
    tile_to_sg = {tid: contract.tiles[tid].spatial_group for tid in val_tiles}
    frozen_ckpt_entries: List[Dict[str, Any]] = []

    for run_label, m_type, c_size in models_to_run:
        for sd in seeds:
            tag = f"{run_label}_s{sd}"
            raw_ckpt = out_dir / "checkpoints" / f"{tag}_raw.pt"
            cal_ckpt = out_dir / "checkpoints" / f"{tag}_calibrated.pt"
            tr_json = out_dir / "receipts" / f"{tag}_train.json"
            cal_json = out_dir / "receipts" / f"{tag}_calibrate.json"
            ev_json = out_dir / "receipts" / f"{tag}_eval_val.json"
            lat_json = out_dir / "receipts" / f"{tag}_latency.json"

            cmd_train(
                argparse.Namespace(
                    split_file=args.split_file,
                    manifest_file=args.manifest_file,
                    allow_synthetic_split=args.allow_synthetic_split,
                    diagnostic_mode=args.diagnostic_mode,
                    global_mesh_dir=args.global_mesh_dir,
                    cache_dir=cache_dir,
                    model_type=m_type,
                    crop_size=c_size,
                    hidden_dim=args.hidden_dim,
                    fit_subset=args.fit_subset,
                    temporal_protocol=args.temporal_protocol,
                    receiver_mode=args.receiver_mode,
                    bbox_margin_m=args.bbox_margin_m,
                    days_per_batch=args.days_per_batch,
                    receiver_chunk_size=args.receiver_chunk_size,
                    max_intra_day_frames=0,
                    selection_rule=selection_rule,
                    max_tiles=args.max_tiles,
                    max_receivers=args.max_receivers,
                    max_candidates=args.max_candidates,
                    num_azimuth_bins=args.num_azimuth_bins,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=1e-4,
                    seed=sd,
                    device=args.device,
                    disable_bf16=False,
                    checkpoint_out=raw_ckpt,
                    output=tr_json,
                    quiet=True,
                )
            )
            cal_res = cmd_calibrate(
                argparse.Namespace(
                    split_file=args.split_file,
                    manifest_file=args.manifest_file,
                    allow_synthetic_split=args.allow_synthetic_split,
                    diagnostic_mode=args.diagnostic_mode,
                    global_mesh_dir=args.global_mesh_dir,
                    cache_dir=cache_dir,
                    checkpoint=raw_ckpt,
                    checkpoint_out=cal_ckpt,
                    temporal_protocol=args.temporal_protocol,
                    receiver_mode=args.receiver_mode,
                    bbox_margin_m=args.bbox_margin_m,
                    days_per_batch=args.days_per_batch,
                    receiver_chunk_size=args.receiver_chunk_size,
                    max_tiles=args.max_tiles,
                    max_receivers=args.max_receivers,
                    max_candidates=args.max_candidates,
                    num_azimuth_bins=args.num_azimuth_bins,
                    device=args.device,
                    output=cal_json,
                    quiet=True,
                )
            )
            ev_res, tile_acc_map = _run_bucket_evaluation(
                argparse.Namespace(
                    split_file=args.split_file,
                    manifest_file=args.manifest_file,
                    allow_synthetic_split=args.allow_synthetic_split,
                    diagnostic_mode=args.diagnostic_mode,
                    global_mesh_dir=args.global_mesh_dir,
                    cache_dir=cache_dir,
                    checkpoint=cal_ckpt,
                    bucket="val_spatial_clean",
                    temporal_protocol=args.temporal_protocol,
                    receiver_mode=args.receiver_mode,
                    bbox_margin_m=args.bbox_margin_m,
                    days_per_batch=args.days_per_batch,
                    receiver_chunk_size=args.receiver_chunk_size,
                    max_intra_day_frames=0,
                    max_tiles=args.max_tiles,
                    max_receivers=args.max_receivers,
                    max_candidates=args.max_candidates,
                    num_azimuth_bins=args.num_azimuth_bins,
                    ledger_file=ledger_path,
                    device=args.device,
                    output=ev_json,
                    quiet=True,
                ),
                contract,
                "val_spatial_clean",
                return_tile_accumulators=True,
            )
            lat_res = cmd_benchmark_latency(
                argparse.Namespace(
                    split_file=args.split_file,
                    manifest_file=args.manifest_file,
                    allow_synthetic_split=args.allow_synthetic_split,
                    diagnostic_mode=args.diagnostic_mode,
                    global_mesh_dir=args.global_mesh_dir,
                    cache_dir=cache_dir,
                    checkpoint=cal_ckpt,
                    bucket="val_spatial_clean",
                    tile_id=None,
                    temporal_protocol=args.temporal_protocol,
                    num_days=1,
                    receiver_mode=args.receiver_mode,
                    bbox_margin_m=args.bbox_margin_m,
                    max_receivers=args.max_receivers,
                    max_candidates=args.max_candidates,
                    num_azimuth_bins=args.num_azimuth_bins,
                    warmup_iters=1,
                    timed_iters=2,
                    physical_timed_iters=1,
                    skip_physical_baseline=False,
                    device=args.device,
                    output=lat_json,
                    quiet=True,
                )
            )

            val_bucket = ev_res["evaluation_buckets_strictly_separated"]["val_spatial_clean"]
            by_model_seed_summaries[run_label].append(val_bucket)
            by_seed_model_tile_accs[sd][run_label] = tile_acc_map
            frozen_ckpt_entries.append(
                {
                    "run_tag": tag,
                    "model_type": m_type,
                    "seed": sd,
                    "checkpoint_path": str(cal_ckpt.resolve()),
                    "checkpoint_sha256": cal_res["calibrated_checkpoint_sha256"],
                }
            )
            model_runs.append(
                {
                    "run_tag": tag,
                    "model_type": m_type,
                    "crop_size": c_size,
                    "seed": sd,
                    "selection_rule": selection_rule,
                    "calibrated_checkpoint": str(cal_ckpt.resolve()),
                    "calibrated_checkpoint_sha256": cal_res["calibrated_checkpoint_sha256"],
                    "train_receipt_sha256": compute_file_sha256(tr_json),
                    "eval_receipt_sha256": compute_file_sha256(ev_json),
                    "latency_receipt_sha256": compute_file_sha256(lat_json),
                    "val_spatial_clean_state_accuracy": val_bucket.get("overall", {}).get("state_accuracy"),
                    "pure_warm_us_per_valid_point_frame": lat_res["neural_pure_warm_gpu_forward"][
                        "us_per_valid_point_frame"
                    ],
                }
            )

    multiseed_table = {
        m_label: aggregate_multiseed_reports(reps)
        for m_label, reps in by_model_seed_summaries.items()
    }

    # Paired bootstrap comparisons (`rind_full_geom` minus each ablation/baseline on seed 0)
    paired_comparisons: Dict[str, Any] = {}
    ref_seed = seeds[0]
    ref_accs = by_seed_model_tile_accs[ref_seed].get("rind_full_geom", {})
    if ref_accs:
        for other_label, _, _ in models_to_run:
            if other_label == "rind_full_geom":
                continue
            other_accs = by_seed_model_tile_accs[ref_seed].get(other_label, {})
            if other_accs:
                paired_comparisons[f"rind_full_geom_minus_{other_label}"] = (
                    paired_bootstrap_model_comparison(
                        ref_accs,
                        other_accs,
                        tile_to_spatial_group=tile_to_sg,
                    )
                )

    speed_json = out_dir / "02_train_speed_benchmark.json"
    speed_res = cmd_benchmark_train_speed(
        argparse.Namespace(
            split_file=args.split_file,
            manifest_file=args.manifest_file,
            allow_synthetic_split=args.allow_synthetic_split,
            diagnostic_mode=args.diagnostic_mode,
            global_mesh_dir=args.global_mesh_dir,
            cache_dir=cache_dir,
            tile_id=None,
            model_type="rind_full_geom",
            crop_size=64,
            hidden_dim=args.hidden_dim,
            temporal_protocol=args.temporal_protocol,
            num_days=1,
            receiver_mode=args.receiver_mode,
            bbox_margin_m=args.bbox_margin_m,
            max_receivers=args.max_receivers,
            max_candidates=args.max_candidates,
            num_azimuth_bins=args.num_azimuth_bins,
            warmup_iters=1,
            timed_iters=3,
            enable_compile=False,
            device=args.device,
            output=speed_json,
            quiet=True,
        )
    )

    inv_json = out_dir / "03_inverse_design.json"
    inv_res = cmd_inverse_design(
        argparse.Namespace(
            split_file=args.split_file,
            manifest_file=args.manifest_file,
            allow_synthetic_split=args.allow_synthetic_split,
            diagnostic_mode=args.diagnostic_mode,
            global_mesh_dir=args.global_mesh_dir,
            bucket="val_spatial_clean",
            tile_id=None,
            checkpoint=Path(frozen_ckpt_entries[0]["checkpoint_path"]),
            optimizer_arm="rind_checkpoint_grad",
            temporal_protocol=args.temporal_protocol,
            num_days=1,
            receiver_mode=args.receiver_mode,
            max_receivers=min(int(args.max_receivers), 64),
            max_candidates=args.max_candidates,
            bbox_margin_m=args.bbox_margin_m,
            editable_radius_m=250.0,
            max_delta_h_m=15.0,
            steps=10,
            lr=0.75,
            disable_volume_conservation=False,
            zero_sum_height=False,
            device=args.device,
            output=inv_json,
            quiet=True,
        )
    )

    freeze_manifest_template = {
        "schema": "rind.blind_unseal_freeze_manifest.v1",
        "benchmark_contract": BENCHMARK_CONTRACT_ID,
        "split_sha256": contract.split_sha256,
        "manifest_sha256": contract.manifest_sha256,
        "evaluator_version": f"{PACKAGE_VERSION}::{FEATURE_SCHEMA_VERSION}",
        "selection_rule": selection_rule,
        "frozen_checkpoints": frozen_ckpt_entries,
        "created_at_utc": _utc_now_iso(),
    }
    freeze_template_path = out_dir / "BLIND_UNSEAL_FREEZE_MANIFEST_TEMPLATE.json"
    _write_json_atomic(freeze_template_path, freeze_manifest_template)

    if contract.blind_open_count != 0:
        raise RuntimeError(
            f"BLIND TEST LOCK VIOLATED DURING DEV SUITE: blind_open_count={contract.blind_open_count}"
        )

    pkg_dir = Path(__file__).resolve().parent
    code_hashes = {p.name: compute_file_sha256(p) for p in sorted(pkg_dir.glob("*.py"))}

    dev_receipt = {
        "schema": "rind.dev_suite_experiment_receipt.v2",
        "receipt_class": receipt_class,
        "benchmark_contract": BENCHMARK_CONTRACT_ID,
        "package_version": PACKAGE_VERSION,
        "feature_version": FEATURE_SCHEMA_VERSION,
        "blind_open_count": contract.blind_open_count,
        "blind_test_status": {
            "bucket": "test_spatial_unexposed",
            "registered_tiles": split_report["counts_by_canonical_split"]["test_spatial_unexposed"],
            "blind_open_count": contract.blind_open_count,
            "result": "NA",
            "note": "Locked during development; unseal once via `unseal-blind-test` after freezing checkpoints.",
        },
        "execution_plan": execution_plan,
        "code_module_sha256": code_hashes,
        "split_verification": {
            "split_sha256": split_report["split_sha256"],
            "manifest_sha256": split_report["manifest_sha256"],
            "counts_by_canonical_split": split_report["counts_by_canonical_split"],
        },
        "val_exposure_ledger_path": str(ledger_path.resolve()),
        "freeze_manifest_template_path": str(freeze_template_path.resolve()),
        "model_runs": model_runs,
        "multiseed_val_summary": multiseed_table,
        "paired_val_comparisons": paired_comparisons,
        "training_speedup_benchmark": {
            "speedup_shard_fp32_vs_raw_fp32": speed_res["speedup_shard_fp32_vs_raw_fp32"],
            "speedup_bf16_vs_shard_fp32": speed_res["speedup_bf16_vs_shard_fp32"],
            "speedup_factor": speed_res["speedup_factor"],
            "fp32_vs_fp32_loss_abs_diff": speed_res["numerical_parity"]["fp32_vs_fp32_loss_abs_diff"],
            "gradient_cosine_similarity": speed_res["numerical_parity"]["gradient_cosine_similarity"],
            "receipt_sha256": compute_file_sha256(speed_json),
        },
        "inverse_design_summary": {
            "optimizer_arm": inv_res["inverse_design_result"]["optimizer_arm"],
            "signed_net_sunlight_delta_fraction": inv_res["inverse_design_result"][
                "signed_net_sunlight_delta_fraction"
            ],
            "building_consistency_verified": inv_res["inverse_design_result"][
                "building_consistency_verified"
            ],
            "receipt_sha256": compute_file_sha256(inv_json),
        },
        "completed_at_utc": _utc_now_iso(),
    }
    receipt_name = (
        "DIAGNOSTIC_DEV_SUITE_RECEIPT.json"
        if receipt_class == "DIAGNOSTIC_NON_PUBLICATION"
        else "DEV_SUITE_EXPERIMENT_RECEIPT_V2.json"
    )
    _write_json_atomic(out_dir / receipt_name, dev_receipt)
    if not args.quiet:
        print(json.dumps(dev_receipt, indent=2))
    return dev_receipt


def _add_streaming_io_args(
    parser: argparse.ArgumentParser,
    *,
    default_days_per_batch: int = 1,
    default_receiver_chunk: int = 256,
) -> None:
    parser.add_argument(
        "--temporal-protocol",
        type=str,
        default="annual_52week",
        help="Temporal schedule (`annual_52week` required for formal publication receipts).",
    )
    parser.add_argument(
        "--receiver-mode",
        type=str,
        default="sampled",
        choices=["sampled", "all_valid"],
        help="Receiver sampling mode (`all_valid` evaluates all eligible static receivers; `sampled` uses deterministic spatial stratification).",
    )
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--max-receivers", type=int, default=256)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--allow-candidate-intervention", action="store_true",help="Explicitly compare inputs using a fixed checkpoint")
    parser.add_argument("--num-azimuth-bins", type=int, default=0, help="0: per-ray source AABB retrieval (default). Positive: legacy fixed sector bank.")
    parser.add_argument("--bbox-margin-m", type=float, default=DEFAULT_SCENE_HORIZON_M)
    parser.add_argument("--days-per-batch", type=int, default=default_days_per_batch)
    parser.add_argument("--receiver-chunk-size", type=int, default=default_receiver_chunk)
    parser.add_argument("--max-intra-day-frames", type=int, default=0)


def _candidate_attention_options(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "film_beta_bound": float(getattr(args, "film_beta_bound", 0.0)),
        "attn_qk_norm": bool(getattr(args, "attn_qk_norm", False)),
        "cand_readout": str(getattr(args, "cand_readout", "attn_only")),
    }


def _add_candidate_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--film-beta-bound", type=float, default=0.0,
                        help="Positive smooth FiLM beta bound; 0 preserves legacy unbounded beta.")
    parser.add_argument("--attn-qk-norm", action="store_true",
                        help="RMS-normalize attention queries and keys before scaled dot product.")
    parser.add_argument("--cand-readout", choices=["attn_only", "attn_plus_lse"], default="attn_only",
                        help="Optional per-face normalized LogSumExp residual before the event gate.")
    parser.add_argument("--cf-subgroup-size", type=int, default=0,
                        help="Independent CF perturbation per receiver subgroup, weighted by subgroup size; 0 disables.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rind",
        description=f"Unified Clean-Room Annual Full-Day RIND Suite ({BENCHMARK_CONTRACT_ID})",
    )
    parser.add_argument("--require-complete-candidates", action="store_true", help="Raise with required width on any truncated source-ray query.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # 1. verify-split
    p_vs = sub.add_parser("verify-split", help="Verify split/manifest/CSV SHA-256, tri-state exposure & blind lock.")
    _add_common_split_args(p_vs)
    p_vs.add_argument("--output", type=Path, default=None)
    p_vs.add_argument("--quiet", action="store_true")
    p_vs.set_defaults(func=cmd_verify_split)

    # 2. inspect-annual-tile
    p_ins = sub.add_parser("inspect-annual-tile", help="Verify 52-week / 18,808-frame / 120s cadence on a tile.")
    _add_common_split_args(p_ins)
    p_ins.add_argument("--tile-id", type=str, default=None)
    p_ins.add_argument("--decode-sample-weeks", action="store_true")
    p_ins.add_argument("--output", type=Path, default=None)
    p_ins.add_argument("--quiet", action="store_true")
    p_ins.set_defaults(func=cmd_inspect_annual_tile)

    # 3. build-mmap-cache
    p_mc = sub.add_parser("build-mmap-cache", help="Build atomic provenance-verified `.npy` mmap shards.")
    _add_common_split_args(p_mc)
    p_mc.add_argument("--cache-dir", type=Path, required=True)
    p_mc.add_argument("--split", type=str, default="fit_8")
    _add_streaming_io_args(p_mc, default_days_per_batch=1)
    p_mc.add_argument("--output", type=Path, default=None)
    p_mc.add_argument("--quiet", action="store_true")
    p_mc.set_defaults(func=cmd_build_mmap_cache)

    # 4. train
    p_tr = sub.add_parser("train", help="Stream-train unified model strictly on `fit` subset.")
    _add_common_split_args(p_tr)
    _add_portable_args(p_tr)
    p_tr.add_argument("--cache-dir", type=Path, required=True)
    p_tr.add_argument("--mainline", action=argparse.BooleanOptionalAction, default=None,
                      help="Defaults on for rind_full_geom. Store this evaluation profile in the checkpoint.")
    p_tr.add_argument("--far-perspective", action=argparse.BooleanOptionalAction, default=None, help="Add learnable angular vertex offsets in the solar tangent frame.")
    p_tr.add_argument("--model-type", type=str, choices=ALL_MODEL_TYPES, required=True)
    p_tr.add_argument("--crop-size", type=int, default=64, choices=[21, 64])
    p_tr.add_argument(
        "--spatial-execution-mode",
        type=str,
        default=None,
        choices=["full_tile_512_halo", "per_receiver_crop"],
        help="Explicit spatial execution mode (`local_convgru_grid` defaults to `per_receiver_crop`; `full_tile_convgru` uses `full_tile_512_halo`).",
    )
    p_tr.add_argument(
        "--convgru-mode",
        type=str,
        default=None,
        choices=["full_tile_512_halo", "per_receiver_crop"],
    )
    p_tr.add_argument("--hidden-dim", type=int, default=96)
    p_tr.add_argument(
        "--jepa-mode",
        type=str,
        default="hybrid",
        choices=list(JEPA_MODES),
    )
    p_tr.add_argument(
        "--ema-target-scope",
        type=str,
        default="head",
        choices=list(EMA_TARGET_SCOPES),
    )
    p_tr.add_argument("--matched-objective",choices=["direct","geometry_mae","geometry_jepa","response_jepa"])
    p_tr.add_argument("--buffer-m", type=float, default=1.5)
    p_tr.add_argument(
        "--mask-mode",
        type=str,
        default="spatial_boundary",
        choices=list(ALL_MASK_MODES),
    )
    p_tr.add_argument("--mask-ratio", type=float, default=0.25)
    p_tr.add_argument("--spatial-block-size", type=int, default=24)
    p_tr.add_argument(
        "--jepa-context-mode",
        type=str,
        default="with_x",
        choices=list(JEPA_CONTEXT_MODES),
    )
    p_tr.add_argument(
        "--reg-mode",
        type=str,
        default="sigreg",
        choices=list(REG_MODES),
    )
    p_tr.add_argument(
        "--transition-mode",
        type=str,
        default="event_aware",
        choices=list(TRANSITION_MODES),
    )
    p_tr.add_argument("--pure-supervised", action="store_true")
    p_tr.add_argument("--label-budget-fraction", type=float, default=1.0)
    p_tr.add_argument("--label-budget-seed", type=int, default=20261112)
    p_tr.add_argument("--counterfactual-weight", type=float, default=0.08)
    _add_candidate_experiment_args(p_tr)
    p_tr.add_argument("--dominant-occluder-dropout-prob", type=float, default=0.25)
    p_tr.add_argument(
        "--attention-mode",
        type=str,
        default="ray_conditioned",
        choices=list(ATTENTION_MODES),
    )
    p_tr.add_argument("--disable-event-gate", action="store_true")
    p_tr.add_argument("--reanchor-every", type=int, default=0)
    p_tr.add_argument("--disable-static-mpnn", action="store_true")
    p_tr.add_argument("--disable-local-knn-readout", action="store_true")
    p_tr.add_argument("--disable-fourier102", action="store_true")
    p_tr.add_argument("--fit-subset", type=str, default="fit_67", choices=list(FIT_SUBSET_SIZES.keys()))
    p_tr.add_argument("--objective-receivers",type=int,default=256,help="Mainline logical receiver support per optimizer step.")
    p_tr.add_argument("--objective-window-frames", type=int, default=0,
                      help="Fixed same-day frames per optimizer update, independent of physical reads. 0 uses legacy loader batches.")
    _add_streaming_io_args(p_tr, default_days_per_batch=1)
    p_tr.add_argument("--selection-rule", type=str, default="fixed_final_epoch")
    p_tr.add_argument("--epochs", type=int, default=5)
    p_tr.add_argument("--max-train-steps", type=int, default=0)
    p_tr.add_argument("--save-every-steps", type=int, default=0)
    p_tr.add_argument("--resume-state-out", type=Path, default=None)
    p_tr.add_argument("--resume-from", type=Path, default=None)
    p_tr.add_argument("--collapse-jsonl", type=Path, default=None)
    p_tr.add_argument("--lr", type=float, default=1.5e-3)
    p_tr.add_argument("--weight-decay", type=float, default=1e-4)
    p_tr.add_argument("--seed", type=int, default=20261111)
    p_tr.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_tr.add_argument("--disable-bf16", action="store_true")
    p_tr.add_argument("--checkpoint-out", type=Path, required=True)
    p_tr.add_argument("--output", type=Path, default=None)
    p_tr.add_argument("--quiet", action="store_true")
    p_tr.set_defaults(func=cmd_train)

    # 5. calibrate
    p_cal = sub.add_parser("calibrate", help="Fit scalar temperature & threshold strictly on `calibration` only.")
    _add_common_split_args(p_cal)
    _add_portable_args(p_cal)
    p_cal.add_argument("--cache-dir", type=Path, required=True)
    p_cal.add_argument("--checkpoint", type=Path, required=True)
    p_cal.add_argument("--checkpoint-out", type=Path, default=None)
    _add_streaming_io_args(p_cal, default_days_per_batch=2)
    p_cal.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_cal.add_argument("--output", type=Path, default=None)
    p_cal.add_argument("--quiet", action="store_true")
    p_cal.set_defaults(func=cmd_calibrate)

    # 6. eval & eval-fullday (development val_spatial_clean)
    for ev_cmd in ("eval", "eval-fullday"):
        p_ev = sub.add_parser(
            ev_cmd,
            help="Stream-evaluate calibrated checkpoint on `val_spatial_clean` (logs exposure ledger; blind test locked).",
        )
        _add_common_split_args(p_ev)
        _add_portable_args(p_ev)
        p_ev.add_argument("--cache-dir", type=Path, required=True)
        p_ev.add_argument("--checkpoint", type=Path, required=True)
        p_ev.add_argument(
            "--bucket",
            type=str,
            default="val_spatial_clean",
            choices=[*CANONICAL_EVAL_BUCKETS, "dev_separated", "all_separated"],
        )
        _add_streaming_io_args(p_ev, default_days_per_batch=2)
        p_ev.add_argument("--eval-stride", type=int, default=1, choices=[1, 2, 4, 8])
        p_ev.add_argument("--inference-mode", type=str, default="recurrent", choices=list(INFERENCE_MODES))
        p_ev.add_argument("--mainline", action=argparse.BooleanOptionalAction, default=None, help="Default to the checkpoint training profile; override diagnostics without changing weights.")
        p_ev.add_argument("--blocker-distance-buckets", action=argparse.BooleanOptionalAction, default=None)
        p_ev.add_argument("--shadow-contours", action=argparse.BooleanOptionalAction, default=None)
        p_ev.add_argument("--allow-uncalibrated", action="store_true")
        p_ev.add_argument("--ledger-file", type=Path, default=None)
        p_ev.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
        p_ev.add_argument("--output", type=Path, default=None)
        p_ev.add_argument("--quiet", action="store_true")
        p_ev.set_defaults(func=cmd_eval_fullday)

    # 7. eval-legacy
    p_leg = sub.add_parser(
        "eval-legacy",
        help="Diagnostic evaluation on `test_spatial_legacy_reassigned` (gated on scratch `fit`-only lineage).",
    )
    _add_common_split_args(p_leg)
    p_leg.add_argument("--cache-dir", type=Path, required=True)
    p_leg.add_argument("--checkpoint", type=Path, required=True)
    _add_streaming_io_args(p_leg, default_days_per_batch=2)
    p_leg.add_argument("--eval-stride", type=int, default=1, choices=[1, 2, 4, 8])
    p_leg.add_argument("--inference-mode", type=str, default="recurrent", choices=list(INFERENCE_MODES))
    p_leg.add_argument("--blocker-distance-buckets", action="store_true")
    p_leg.add_argument("--shadow-contours", action="store_true")
    p_leg.add_argument("--allow-uncalibrated", action="store_true")
    p_leg.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_leg.add_argument("--output", type=Path, default=None)
    p_leg.add_argument("--quiet", action="store_true")
    p_leg.set_defaults(func=cmd_eval_legacy)

    # 8. unseal-blind-test
    p_ub = sub.add_parser(
        "unseal-blind-test",
        help="One-shot blind evaluation on `test_spatial_unexposed` requiring frozen pre-registration manifest.",
    )
    _add_common_split_args(p_ub)
    p_ub.add_argument("--cache-dir", type=Path, required=True)
    p_ub.add_argument("--checkpoint", type=Path, required=True)
    p_ub.add_argument("--freeze-manifest", type=Path, required=True)
    _add_streaming_io_args(p_ub, default_days_per_batch=2)
    p_ub.add_argument("--inference-mode", type=str, default="recurrent", choices=list(INFERENCE_MODES))
    p_ub.add_argument("--blocker-distance-buckets", action="store_true")
    p_ub.add_argument("--shadow-contours", action="store_true")
    p_ub.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_ub.add_argument("--output", type=Path, default=None)
    p_ub.add_argument("--quiet", action="store_true")
    p_ub.set_defaults(func=cmd_unseal_blind_test)

    # 9. benchmark-latency
    p_lat = sub.add_parser(
        "benchmark-latency",
        help="Matched-domain cold, cached, pure-warm GPU & exact 4km physical ray-trace latency benchmark.",
    )
    _add_common_split_args(p_lat)
    p_lat.add_argument("--cache-dir", type=Path, required=True)
    p_lat.add_argument("--checkpoint", type=Path, required=True)
    p_lat.add_argument("--bucket", type=str, default="val_spatial_clean", choices=list(CANONICAL_EVAL_BUCKETS))
    p_lat.add_argument("--tile-id", type=str, default=None)
    p_lat.add_argument("--temporal-protocol", type=str, default="annual_52week")
    p_lat.add_argument("--num-days", type=int, default=1)
    p_lat.add_argument("--receiver-mode", type=str, default="sampled", choices=["sampled", "all_valid"])
    p_lat.add_argument("--max-receivers", type=int, default=256)
    p_lat.add_argument("--max-candidates", type=int, default=32)
    p_lat.add_argument("--num-azimuth-bins", type=int, default=0, help="0: per-ray source AABB retrieval (default). Positive: legacy fixed sector bank.")
    p_lat.add_argument("--bbox-margin-m", type=float, default=DEFAULT_SCENE_HORIZON_M)
    p_lat.add_argument("--warmup-iters", type=int, default=2)
    p_lat.add_argument("--timed-iters", type=int, default=5)
    p_lat.add_argument("--physical-timed-iters", type=int, default=1)
    p_lat.add_argument("--skip-physical-baseline", action="store_true")
    p_lat.add_argument("--allow-uncalibrated", action="store_true")
    p_lat.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_lat.add_argument("--output", type=Path, default=None)
    p_lat.add_argument("--quiet", action="store_true")
    p_lat.set_defaults(func=cmd_benchmark_latency)

    # 10. benchmark-train-speed
    p_sp = sub.add_parser(
        "benchmark-train-speed",
        help="3-arm matched training speedup benchmark (`baseline_raw_fp32`, `fast_shard_fp32`, `fast_shard_bf16_mixed`).",
    )
    _add_common_split_args(p_sp)
    p_sp.add_argument("--cache-dir", type=Path, required=True)
    p_sp.add_argument("--tile-id", type=str, default=None)
    p_sp.add_argument("--model-type", type=str, default="rind_full_geom", choices=ALL_MODEL_TYPES)
    p_sp.add_argument("--crop-size", type=int, default=64, choices=[21, 64])
    p_sp.add_argument("--hidden-dim", type=int, default=96)
    _add_candidate_experiment_args(p_sp)
    p_sp.add_argument("--temporal-protocol", type=str, default="annual_52week")
    p_sp.add_argument("--num-days", type=int, default=2)
    p_sp.add_argument("--receiver-mode", type=str, default="sampled", choices=["sampled", "all_valid"])
    p_sp.add_argument("--max-receivers", type=int, default=256)
    p_sp.add_argument("--max-candidates", type=int, default=32)
    p_sp.add_argument("--num-azimuth-bins", type=int, default=0, help="0: per-ray source AABB retrieval (default). Positive: legacy fixed sector bank.")
    p_sp.add_argument("--bbox-margin-m", type=float, default=DEFAULT_SCENE_HORIZON_M)
    p_sp.add_argument("--warmup-iters", type=int, default=2)
    p_sp.add_argument("--timed-iters", type=int, default=5)
    p_sp.add_argument("--enable-compile", action="store_true")
    p_sp.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_sp.add_argument("--output", type=Path, default=None)
    p_sp.add_argument("--quiet", action="store_true")
    p_sp.set_defaults(func=cmd_benchmark_train_speed)

    # 11. inverse-design
    p_inv = sub.add_parser(
        "inverse-design",
        help="Full 4km `ExactSceneBVH` building-ID consistent height optimization across 3 optimizer arms.",
    )
    _add_common_split_args(p_inv)
    p_inv.add_argument("--bucket", type=str, default="val_spatial_clean", choices=list(CANONICAL_ALL_BUCKETS))
    p_inv.add_argument("--tile-id", type=str, default=None)
    p_inv.add_argument("--tile-ids", type=str, default=None, help="Optional comma-separated tile IDs for multi-tile aggregation.")
    p_inv.add_argument("--checkpoint", type=Path, default=None)
    p_inv.add_argument(
        "--optimizer-arm",
        type=str,
        default="analytic_diff_surrogate",
        choices=["rind_checkpoint_grad", "analytic_diff_surrogate", "blackbox_zero_order"],
    )
    p_inv.add_argument("--temporal-protocol", type=str, default="annual_52week")
    p_inv.add_argument("--num-days", type=int, default=1)
    p_inv.add_argument("--receiver-mode", type=str, default="sampled", choices=["sampled", "all_valid"])
    p_inv.add_argument("--max-receivers", type=int, default=64)
    p_inv.add_argument("--max-candidates", type=int, default=32)
    p_inv.add_argument("--bbox-margin-m", type=float, default=DEFAULT_SCENE_HORIZON_M)
    p_inv.add_argument("--editable-radius-m", type=float, default=250.0)
    p_inv.add_argument("--max-delta-h-m", type=float, default=15.0)
    p_inv.add_argument("--steps", type=int, default=30)
    p_inv.add_argument("--lr", type=float, default=0.75)
    p_inv.add_argument("--max-oracle-queries", type=int, default=None)
    p_inv.add_argument("--max-wall-clock-s", type=float, default=None)
    p_inv.add_argument("--allow-nonclosed-surface-proxy", action="store_true")
    p_inv.add_argument("--allow-uncalibrated", action="store_true")
    p_inv.add_argument("--disable-volume-conservation", action="store_true")
    p_inv.add_argument("--zero-sum-height", action="store_true")
    p_inv.add_argument("--device", type=str, default="cpu")
    p_inv.add_argument("--output", type=Path, default=None)
    p_inv.add_argument("--quiet", action="store_true")
    p_inv.set_defaults(func=cmd_inverse_design)

    # 12. infer-building-height
    p_ibh = sub.add_parser(
        "infer-building-height",
        help="Bayesian posterior height inference for unknown buildings under noisy shadow observations (`P1`).",
    )
    _add_common_split_args(p_ibh)
    p_ibh.add_argument("--synthetic-benchmark", action="store_true")
    p_ibh.add_argument("--bucket", type=str, default="val_spatial_clean", choices=list(CANONICAL_ALL_BUCKETS))
    p_ibh.add_argument("--tile-id", type=str, default=None)
    p_ibh.add_argument("--checkpoint", type=Path, default=None)
    p_ibh.add_argument("--temporal-protocol", type=str, default="annual_52week")
    p_ibh.add_argument("--num-days", type=int, default=1)
    p_ibh.add_argument("--receiver-mode", type=str, default="sampled", choices=["sampled", "all_valid"])
    p_ibh.add_argument("--max-receivers", type=int, default=32)
    p_ibh.add_argument("--max-candidates", type=int, default=32)
    p_ibh.add_argument("--bbox-margin-m", type=float, default=DEFAULT_SCENE_HORIZON_M)
    p_ibh.add_argument("--unknown-building-ids",nargs="+",type=int)
    p_ibh.add_argument("--receiver-elevation-public",action="store_true")
    p_ibh.add_argument("--heldout-frame-indices",nargs="+",type=int,default=[])
    p_ibh.add_argument("--max-unknown-buildings", type=int, default=2)
    p_ibh.add_argument("--height-grid-min-m", type=float, default=2.0)
    p_ibh.add_argument("--height-grid-max-m", type=float, default=50.0)
    p_ibh.add_argument("--num-grid-hypotheses", type=int, default=25)
    p_ibh.add_argument("--sensor-noise", type=float, default=0.05)
    p_ibh.add_argument("--initial-queries", type=int, default=6)
    p_ibh.add_argument("--active-rounds", type=int, default=4)
    p_ibh.add_argument("--queries-per-round", type=int, default=6)
    p_ibh.add_argument("--seed", type=int, default=20261112)
    p_ibh.add_argument("--device", type=str, default="cpu")
    p_ibh.add_argument("--output", type=Path, default=None)
    p_ibh.add_argument("--quiet", action="store_true")
    p_ibh.set_defaults(func=cmd_infer_building_height)

    # 13. eval-repr-sufficiency
    p_rs = sub.add_parser(
        "eval-repr-sufficiency",
        help="Evaluate representation sufficiency under driver shifts & nonadditive unseen edit combinations.",
    )
    _add_common_split_args(p_rs)
    p_rs.add_argument("--synthetic-benchmark", action="store_true")
    p_rs.add_argument("--bucket", type=str, default="val_spatial_clean", choices=list(CANONICAL_ALL_BUCKETS))
    p_rs.add_argument("--tile-id", type=str, default=None)
    p_rs.add_argument("--checkpoint", type=Path, default=None)
    p_rs.add_argument("--temporal-protocol", type=str, default="annual_52week")
    p_rs.add_argument("--num-days", type=int, default=1)
    p_rs.add_argument("--receiver-mode", type=str, default="sampled", choices=["sampled", "all_valid"])
    p_rs.add_argument("--max-receivers", type=int, default=48)
    p_rs.add_argument("--max-candidates", type=int, default=32)
    p_rs.add_argument("--bbox-margin-m", type=float, default=DEFAULT_SCENE_HORIZON_M)
    p_rs.add_argument("--edit-drop-fraction", type=float, default=0.90)
    p_rs.add_argument("--max-building-pairs", type=int, default=4)
    p_rs.add_argument("--device", type=str, default="cpu")
    p_rs.add_argument("--output", type=Path, default=None)
    p_rs.add_argument("--quiet", action="store_true")
    p_rs.set_defaults(func=cmd_eval_repr_sufficiency)

    # 12. run-dev-suite (and run-master-suite alias)
    for suite_cmd in ("run-dev-suite", "run-master-suite"):
        p_ms = sub.add_parser(
            suite_cmd,
            help="Run strictly development-only (`fit` -> `calibration` -> `val_spatial_clean`) multi-seed suite.",
        )
        _add_common_split_args(p_ms)
        p_ms.add_argument("--work-dir", type=Path, required=True)
        p_ms.add_argument("--fit-subset", type=str, default="fit_67")
        _add_streaming_io_args(p_ms, default_days_per_batch=1)
        p_ms.add_argument("--selection-rule", type=str, default="fixed_final_epoch")
        p_ms.add_argument("--seeds", type=str, default="20261111,20261112,20261113")
        p_ms.add_argument("--hidden-dim", type=int, default=96)
        p_ms.add_argument("--epochs", type=int, default=5)
        p_ms.add_argument("--lr", type=float, default=1.5e-3)
        p_ms.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
        p_ms.add_argument("--quiet", action="store_true")
        p_ms.set_defaults(func=cmd_run_dev_suite)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    from .ray_candidates import require_complete
    token = require_complete.set(args.require_complete_candidates)
    try:
        args.func(args)
    finally:
        require_complete.reset(token)
    return 0
