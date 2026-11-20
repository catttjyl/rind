"""Authoritative Spatial Split Contract, Tri-State Exposure Firewall & Blind-Test Lock
(`MANHATTAN_4445_ANNUAL_18808_V1`).

Enforces:
- `fit` (`fit_8 ⊂ fit_17 ⊂ fit_34 ⊂ fit_67 ⊆ fit_all`): ONLY `authoritative_role == "fit"`
  excluding `ADDITIONAL_FIT_HOLDOUT` (`{"core_fi-0029-fj-0043"}`).
- `calibration`: ONLY `authoritative_role == "calibration"` (post-training scalar temperature /
  threshold / event-gate tuning ONLY; strictly blocked from gradient updates and normalizers).
- `val_spatial_clean`: ONLY `authoritative_role == "val"` (development selection split; logged in
  append-only exposure ledger).
- `test_spatial_unexposed`: Primary 100% blind spatial test (`authoritative_role == "test"` AND
  `historical_exposure_status == "unexposed"`). Locked during development (`blind_open_count == 0`);
  may ONLY be unsealed via `unseal_freeze_manifest` after freezing checkpoints, calibration SHAs,
  evaluator, and selection rule.
- `test_spatial_legacy_reassigned`: Diagnostic archive (`authoritative_role == "test"` AND
  `historical_exposure_status == "exposed"`, gated on `trained_from_scratch_fit_only == True`).
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, FrozenSet, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import BENCHMARK_CONTRACT_ID, FEATURE_SCHEMA_VERSION

EXPECTED_MANHATTAN_4445_SPLIT_SHA256 = (
    "925a3d58f93899ba19b0eeaecc2c88eecdff6f03629c815b5ccd04e2107e97ed"
)
EXPECTED_SNAPSHOT_V1_MANIFEST_SHA256 = (
    "a0e37fc76ced60b684cdf35c90f9f5883bd9aa6acf6792c231cabf3c84b82fa1"
)
EXPECTED_SNAPSHOT_V1_CSV_SHA256 = (
    "c0723779c51b8f0f186be5409d6b8e9a0f021733637435976ae9dadcd98383cb"
)
EXPECTED_UNIFIED_SPEC_V1_SHA256 = (
    "31394067eb64a8761e51e507fb9d5a3f52039a3dab06f8c17f5758491a0386fb"
)
ADDITIONAL_FIT_HOLDOUT: FrozenSet[str] = frozenset({"core_fi-0029-fj-0043"})
EXPECTED_ANNUAL_FRAMES_PER_TILE = 18808
EXPECTED_ANNUAL_WEEKS_PER_TILE = 52

VALID_EXPOSURE_STATUSES: FrozenSet[str] = frozenset({"exposed", "unexposed", "unknown"})

FIT_SUBSET_SIZES: Mapping[str, Optional[int]] = {
    "fit_8": 8,
    "fit_17": 17,
    "fit_34": 34,
    "fit_67": 67,
    "fit_all": None,
    "fit": None,
}

CANONICAL_EVAL_BUCKETS: Tuple[str, ...] = (
    "val_spatial_clean",
    "test_spatial_unexposed",
    "test_spatial_legacy_reassigned",
)

CANONICAL_ALL_BUCKETS: Tuple[str, ...] = (
    "fit",
    "student_holdout",
    "calibration",
    "val_spatial_clean",
    "test_spatial_unexposed",
    "test_spatial_legacy_reassigned",
)

# Relative locations inside the World-Model repository root
REL_SPLIT_JSON = Path(
    "03_训练与实验/接手准备/20261030-独立TRAIN输入验证/MANHATTAN_4445_SPLIT_V1.json"
)
REL_ANNUAL_MANIFEST_JSON = Path(
    "01_数据生成与真值/工作过程/年度全天候标准数据集_20261111/ANNUAL_FULLDAY_18808_MANIFEST_V1.json"
)
REL_ANNUAL_CATALOG_CSV = Path(
    "01_数据生成与真值/交付结果/数据来源与现行数据/标准年度全天候地块目录_V1.csv"
)


@contextlib.contextmanager
def exclusive_file_lock(lock_path: Path | str) -> Iterator[None]:
    """Cross-platform inter-process exclusive advisory file lock (`fcntl` on POSIX, `msvcrt` on Windows)."""
    lp = Path(lock_path)
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("a+b") as fh:
        if os.name == "nt":
            import msvcrt  # type: ignore

            fh.seek(0)
            if lp.stat().st_size == 0:
                fh.write(b"\x00")
                fh.flush()
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def atomic_update_json_file(
    target_path: Path | str,
    updater_fn: Any,
    *,
    lock_path: Optional[Path | str] = None,
    default_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """Read-modify-write a JSON document atomically under an inter-process exclusive lock."""
    tp = Path(target_path)
    tp.parent.mkdir(parents=True, exist_ok=True)
    lp = Path(lock_path) if lock_path is not None else tp.with_name(f".{tp.name}.lock")
    with exclusive_file_lock(lp):
        current: Optional[Dict[str, Any]] = None
        if tp.exists():
            raw_txt = tp.read_text(encoding="utf-8").strip()
            if raw_txt:
                loaded = json.loads(raw_txt)
                if isinstance(loaded, dict):
                    current = loaded
        if current is None and default_factory is not None:
            current = dict(default_factory())
        updated = dict(updater_fn(current))
        payload = json.dumps(updated, indent=2, ensure_ascii=False)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{tp.name}.tmp.", dir=str(tp.parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f_tmp:
                f_tmp.write(payload)
                f_tmp.flush()
                os.fsync(f_tmp.fileno())
            os.replace(str(tmp_path), str(tp))
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise
        return updated


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _stable_rank_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}::{value}".encode("utf-8")).hexdigest()


def discover_world_model_root(start: Optional[Path] = None) -> Optional[Path]:
    env_root = os.environ.get("RIND_WORLD_MODEL_ROOT")
    if env_root:
        candidate = Path(env_root)
        if candidate.exists():
            return candidate.resolve()

    candidates: List[Path] = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path(__file__).resolve())
    candidates.append(Path.cwd().resolve())

    bundle_root = Path(__file__).resolve().parents[1]
    candidates.append(bundle_root)

    for base in candidates:
        for parent in [base, *base.parents]:
            if (parent / REL_SPLIT_JSON).exists():
                return parent
    return None


def resolve_default_split_and_manifest(
    split_path: Optional[Path | str] = None,
    manifest_path: Optional[Path | str] = None,
) -> Tuple[Path, Path]:
    sp_env = os.environ.get("RIND_SPLIT_JSON")
    mf_env = os.environ.get("RIND_MANIFEST_JSON")

    bundle_meta = Path(__file__).resolve().parents[1] / "meta"
    root = discover_world_model_root()
    resolved_split = (
        Path(split_path)
        if split_path is not None
        else Path(sp_env)
        if sp_env
        else (root / REL_SPLIT_JSON if root is not None else (bundle_meta / "split.json" if (bundle_meta / "split.json").exists() else REL_SPLIT_JSON))
    )
    resolved_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else Path(mf_env)
        if mf_env
        else (root / REL_ANNUAL_MANIFEST_JSON if root is not None else (bundle_meta / "manifest.json" if (bundle_meta / "manifest.json").exists() else REL_ANNUAL_MANIFEST_JSON))
    )
    return resolved_split, resolved_manifest


def resolve_tristate_exposure_status(
    tile_id: str,
    raw_tile_entry: Mapping[str, Any],
) -> str:
    """Resolve tri-state `historical_exposure_status` (`exposed` | `unexposed` | `unknown`).

    When explicit `historical_exposure_status` is absent, boolean `False` (`has_historical_exposure=False`
    or `historical_*=False`) is NEVER coerced to `unexposed` — it resolves strictly to `unknown`.
    """
    explicit_tri: Optional[str] = None
    if "historical_exposure_status" in raw_tile_entry and raw_tile_entry["historical_exposure_status"] is not None:
        val = str(raw_tile_entry["historical_exposure_status"]).strip().lower()
        if val not in VALID_EXPOSURE_STATUSES:
            raise ValueError(
                f"Tile {tile_id} has invalid historical_exposure_status={val!r}; "
                f"must be one of {sorted(VALID_EXPOSURE_STATUSES)}."
            )
        explicit_tri = val

    bool_signals: List[Tuple[str, str]] = []
    if "zero_historical_exposure" in raw_tile_entry and isinstance(
        raw_tile_entry["zero_historical_exposure"], bool
    ):
        bool_signals.append(
            (
                "zero_historical_exposure",
                "unexposed" if raw_tile_entry["zero_historical_exposure"] else "exposed",
            )
        )
    if "has_historical_exposure" in raw_tile_entry and isinstance(
        raw_tile_entry["has_historical_exposure"], bool
    ):
        bool_signals.append(
            (
                "has_historical_exposure",
                "exposed" if raw_tile_entry["has_historical_exposure"] else "unknown",
            )
        )
    for key, val in raw_tile_entry.items():
        if key in ("historical_exposure_status", "has_historical_exposure", "zero_historical_exposure"):
            continue
        if key.startswith("historical_") and isinstance(val, bool):
            bool_signals.append((key, "exposed" if val else "unknown"))

    if explicit_tri is None:
        positive_exposed = [imp for _, imp in bool_signals if imp == "exposed"]
        positive_zero = [
            imp for k, imp in bool_signals if k == "zero_historical_exposure" and imp == "unexposed"
        ]
        if positive_exposed and positive_zero:
            raise ValueError(
                f"Exposure conflict on tile {tile_id}: contradictory boolean exposure signals {bool_signals}."
            )
        if positive_exposed:
            status = "exposed"
        elif positive_zero:
            status = "unexposed"
        else:
            return "unknown"
    else:
        status = explicit_tri

    for key_name, implied in bool_signals:
        if implied == "unknown":
            # `has_historical_exposure=False` is compatible with explicit `unexposed` or `unknown`,
            # but contradicts explicit `exposed`!
            if status == "exposed":
                raise ValueError(
                    f"Exposure conflict on tile {tile_id}: status='exposed' contradicts "
                    f"{key_name}=False."
                )
            continue
        if status != "unknown" and implied != status:
            raise ValueError(
                f"Exposure conflict on tile {tile_id}: status={status!r} contradicts "
                f"{key_name} (implies {implied!r})."
            )
    return status


def extract_and_validate_exposure_status(
    tile_id: str,
    raw_tile_entry: Mapping[str, Any],
    *,
    allow_unknown: bool = False,
) -> str:
    """Resolve & validate tri-state `historical_exposure_status` (`exposed` | `unexposed` | `unknown`).

    NEVER defaults missing exposure metadata or `has_historical_exposure=False` to `unexposed`!
    Raises `ValueError` on missing/`"unknown"` (unless `allow_unknown=True`), invalid, or contradictory fields.
    """
    status = resolve_tristate_exposure_status(tile_id, raw_tile_entry)
    if status == "unknown":
        if allow_unknown:
            return "unknown"
        raise ValueError(
            f"Tile {tile_id} resolved to historical_exposure_status='unknown' "
            f"(boolean False -> 'unexposed' fallback is strictly prohibited), which is blocked "
            f"from formal split registration."
        )

    raw_role = str(raw_tile_entry.get("canonical_role", "")).strip()
    legacy_alias = str(raw_tile_entry.get("legacy_partition_alias", "")).strip()
    for role_field_name, role_val in (("canonical_role", raw_role), ("legacy_partition_alias", legacy_alias)):
        if not role_val:
            continue
        if "unexposed" in role_val and status != "unexposed":
            raise ValueError(
                f"Exposure conflict on tile {tile_id}: {role_field_name}={role_val!r} "
                f"requires status='unexposed', got {status!r}."
            )
        if "reassigned" in role_val and status != "exposed":
            raise ValueError(
                f"Exposure conflict on tile {tile_id}: {role_field_name}={role_val!r} "
                f"requires status='exposed', got {status!r}."
            )

    source_batch = str(raw_tile_entry.get("source_batch", "")).strip()
    if source_batch in ("r05-3", "expansion", "fresh_unexposed") and status != "unexposed":
        raise ValueError(
            f"Exposure conflict on tile {tile_id}: source_batch={source_batch!r} "
            f"contradicts status={status!r}."
        )
    if source_batch in ("r05aa",) and status != "exposed":
        raise ValueError(
            f"Exposure conflict on tile {tile_id}: source_batch={source_batch!r} "
            f"contradicts status={status!r}."
        )

    return status


@dataclasses.dataclass(frozen=True)
class TileSplitRecord:
    tile_id: str
    authoritative_role: str  # "fit" | "calibration" | "val" | "test"
    canonical_split: str  # one of CANONICAL_ALL_BUCKETS
    spatial_group: str
    block_row: int
    block_col: int
    fi: int
    fj: int
    historical_exposure_status: str  # "exposed" | "unexposed"
    has_historical_exposure: bool
    source_batch: str
    source_core_sha256: str
    annual_frames: int
    annual_weeks: int
    win_data_dir: str
    wsl_data_dir: str
    win_reader_entry: str
    wsl_reader_entry: str


@dataclasses.dataclass(frozen=True)
class FitFirewallToken:
    """Capability token required by gradient trainers & running feature normalizers."""

    benchmark_contract: str
    split_sha256: str
    manifest_sha256: str
    subset_name: str
    allowed_tile_ids: FrozenSet[str]
    _signature: str

    def verify_tiles(
        self,
        tile_ids: Iterable[str],
        expected_split_sha256: str,
        expected_manifest_sha256: Optional[str] = None,
    ) -> None:
        if self.benchmark_contract != BENCHMARK_CONTRACT_ID:
            raise RuntimeError(
                f"Firewall benchmark contract mismatch: {self.benchmark_contract} != {BENCHMARK_CONTRACT_ID}"
            )
        if self.split_sha256 != expected_split_sha256:
            raise RuntimeError(
                f"Firewall split SHA-256 mismatch: {self.split_sha256} != {expected_split_sha256}"
            )
        if expected_manifest_sha256 is not None and self.manifest_sha256 != expected_manifest_sha256:
            raise RuntimeError(
                f"Firewall manifest SHA-256 mismatch: {self.manifest_sha256} != {expected_manifest_sha256}"
            )
        expected_sig = _compute_firewall_signature(
            self.split_sha256, self.manifest_sha256, self.subset_name, self.allowed_tile_ids
        )
        if self._signature != expected_sig:
            raise RuntimeError("Forged or corrupted FitFirewallToken signature detected.")
        checked = list(tile_ids)
        if not checked:
            raise RuntimeError("Empty tile_ids passed to FitFirewallToken.verify_tiles.")
        bad = [tid for tid in checked if tid not in self.allowed_tile_ids]
        if bad:
            raise RuntimeError(
                f"SPLIT FIREWALL VIOLATION: non-fit or out-of-subset tiles {bad[:5]} "
                f"attempted to enter fit-only computation (subset={self.subset_name})."
            )


def _compute_firewall_signature(
    split_sha256: str,
    manifest_sha256: str,
    subset_name: str,
    allowed_tile_ids: FrozenSet[str],
) -> str:
    joined = ",".join(sorted(allowed_tile_ids))
    payload = f"{BENCHMARK_CONTRACT_ID}|{split_sha256}|{manifest_sha256}|{subset_name}|{joined}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SplitContract:
    """Enforces `MANHATTAN_4445_ANNUAL_18808_V1` split integrity, exposure tri-state, and blind lock."""

    def __init__(
        self,
        split_path: Optional[Path | str] = None,
        manifest_path: Optional[Path | str] = None,
        *,
        enforce_pinned_split_sha256: bool = True,
        blind_ledger_path: Optional[Path | str] = None,
    ) -> None:
        resolved_split, resolved_manifest = resolve_default_split_and_manifest(
            split_path, manifest_path
        )
        if not resolved_split.exists():
            raise FileNotFoundError(f"Authoritative split JSON not found: {resolved_split}")
        if not resolved_manifest.exists():
            raise FileNotFoundError(f"Annual 18808 manifest JSON not found: {resolved_manifest}")

        self.split_path = resolved_split.resolve()
        self.manifest_path = resolved_manifest.resolve()
        self.split_sha256 = compute_file_sha256(self.split_path)
        self.manifest_sha256 = compute_file_sha256(self.manifest_path)
        self.enforce_pinned_split_sha256 = bool(enforce_pinned_split_sha256)
        env_ledger = os.environ.get("RIND_BLIND_UNSEAL_LEDGER")
        self.blind_ledger_path = (
            Path(blind_ledger_path).resolve()
            if blind_ledger_path is not None
            else Path(env_ledger).resolve()
            if env_ledger
            else (self.manifest_path.parent / ".blind_unseal_ledger.jsonl").resolve()
        )
        self._blind_unseal_records: List[Dict[str, Any]] = self._load_persistent_blind_ledger()
        self.blind_open_count: int = len(self._blind_unseal_records)
        self._session_blind_unsealed_tiles: set[str] = set()

        if enforce_pinned_split_sha256 and self.split_sha256 != EXPECTED_MANHATTAN_4445_SPLIT_SHA256:
            raise RuntimeError(
                f"MANHATTAN_4445_SPLIT_V1.json SHA-256 mismatch: got {self.split_sha256}, "
                f"expected pinned {EXPECTED_MANHATTAN_4445_SPLIT_SHA256}"
            )

        raw_split = json.loads(self.split_path.read_text(encoding="utf-8"))
        raw_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.fixture_kind = str(raw_manifest.get("fixture_kind", "formal_annual_manifest"))
        self.is_synthetic_fixture = bool(
            raw_manifest.get("synthetic") is True or self.fixture_kind != "formal_annual_manifest"
        )

        manifest_split_sha = (
            raw_manifest.get("authoritative_split_sha256")
            or raw_manifest.get("authoritative_split_contract", {}).get("sha256")
        )
        if manifest_split_sha and manifest_split_sha != self.split_sha256:
            raise RuntimeError(
                f"Annual manifest authoritative_split_sha256 ({manifest_split_sha}) "
                f"does not match split file SHA-256 ({self.split_sha256})."
            )

        raw_4445_tiles = self._normalize_tiles_collection(raw_split.get("tiles"))
        if not raw_4445_tiles:
            raise RuntimeError(f"Empty `tiles` collection in {self.split_path}")

        # Verify 1km x 1km spatial_group disjointness across the entire 4445-tile universe
        group_to_role: Dict[str, str] = {}
        for tid, info in raw_4445_tiles.items():
            role = self._extract_authoritative_4way_role(info)
            sg = str(info["spatial_group"])
            prev = group_to_role.get(sg)
            if prev is not None and prev != role:
                raise RuntimeError(
                    f"1km spatial_group overlap in split file for {sg}: {prev} vs {role} (tile {tid})"
                )
            group_to_role[sg] = role

        self.universe_tile_roles: Dict[str, str] = {
            tid: (
                "student_holdout"
                if tid in ADDITIONAL_FIT_HOLDOUT
                else self._extract_authoritative_4way_role(info)
            )
            for tid, info in raw_4445_tiles.items()
        }
        self.universe_spatial_groups: Dict[str, str] = {
            tid: str(info["spatial_group"]) for tid, info in raw_4445_tiles.items()
        }

        # Optional top-level partitions cross-check
        declared_partition_by_tile: Dict[str, str] = {}
        raw_partitions = raw_manifest.get("partitions")
        if isinstance(raw_partitions, Mapping):
            for part_name, t_list in raw_partitions.items():
                if part_name not in CANONICAL_ALL_BUCKETS:
                    raise ValueError(
                        f"Unknown partition key {part_name!r} in manifest `partitions`."
                    )
                if not isinstance(t_list, list):
                    raise ValueError(f"Manifest partition {part_name!r} must be a list of tile_ids.")
                for tid in t_list:
                    tid_s = str(tid)
                    if tid_s in declared_partition_by_tile:
                        raise RuntimeError(
                            f"Tile {tid_s} appears in multiple manifest `partitions`: "
                            f"{declared_partition_by_tile[tid_s]} and {part_name}"
                        )
                    declared_partition_by_tile[tid_s] = part_name

        # Parse ready annual 18,808-frame tiles
        self.tiles: Dict[str, TileSplitRecord] = {}
        raw_ready_tiles = self._normalize_tiles_collection(raw_manifest.get("tiles"))
        for tid, entry in sorted(raw_ready_tiles.items()):
            if tid not in raw_4445_tiles:
                raise RuntimeError(
                    f"Manifest tile {tid} does not exist in MANHATTAN_4445_SPLIT_V1.json!"
                )
            u_info = raw_4445_tiles[tid]
            auth_role = self._extract_authoritative_4way_role(u_info)
            entry_auth_role = self._extract_manifest_4way_role(entry, fallback=auth_role)
            if entry_auth_role != auth_role:
                raise RuntimeError(
                    f"Role conflict for {tid}: split file has {auth_role}, manifest has {entry_auth_role}"
                )
            sg = str(u_info["spatial_group"])
            if "spatial_group" in entry and str(entry["spatial_group"]) != sg:
                raise RuntimeError(
                    f"spatial_group conflict for {tid}: {sg} vs {entry['spatial_group']}"
                )

            exp_status = extract_and_validate_exposure_status(tid, entry)
            has_hist = exp_status == "exposed"
            canonical_split = self._map_to_canonical_split(tid, auth_role, exp_status)

            # Verify canonical_role if explicitly present in manifest entry
            raw_c_role = str(entry.get("canonical_role", "")).strip()
            if raw_c_role in CANONICAL_ALL_BUCKETS and raw_c_role != canonical_split:
                raise ValueError(
                    f"Canonical split conflict on {tid}: manifest canonical_role={raw_c_role!r} "
                    f"!= authoritative derived {canonical_split!r}"
                )
            if declared_partition_by_tile and declared_partition_by_tile.get(tid) != canonical_split:
                raise ValueError(
                    f"Manifest `partitions` conflict on {tid}: partitions says "
                    f"{declared_partition_by_tile.get(tid)!r}, derived {canonical_split!r}"
                )

            annual_frames = int(entry.get("annual_frames", EXPECTED_ANNUAL_FRAMES_PER_TILE))
            annual_weeks = int(entry.get("annual_weeks", EXPECTED_ANNUAL_WEEKS_PER_TILE))
            if annual_frames != EXPECTED_ANNUAL_FRAMES_PER_TILE:
                raise RuntimeError(
                    f"Tile {tid} has {annual_frames} annual_frames != {EXPECTED_ANNUAL_FRAMES_PER_TILE}"
                )
            if annual_weeks != EXPECTED_ANNUAL_WEEKS_PER_TILE:
                raise RuntimeError(
                    f"Tile {tid} has {annual_weeks} annual_weeks != {EXPECTED_ANNUAL_WEEKS_PER_TILE}"
                )

            source_core_sha = str(entry.get("source_core_sha256", "")).strip()
            if not self.is_synthetic_fixture and not source_core_sha:
                raise ValueError(
                    f"Formal manifest tile {tid} is missing required `source_core_sha256`."
                )

            t_idx = u_info.get("tile_index") or entry.get("tile_index") or [0, 0]
            fi_val = int(u_info.get("fi", t_idx[0] if len(t_idx) > 0 else 0))
            fj_val = int(u_info.get("fj", t_idx[1] if len(t_idx) > 1 else 0))

            self.tiles[tid] = TileSplitRecord(
                tile_id=tid,
                authoritative_role=auth_role,
                canonical_split=canonical_split,
                spatial_group=sg,
                block_row=int(u_info.get("block_row", fi_val // 10)),
                block_col=int(u_info.get("block_col", fj_val // 10)),
                fi=fi_val,
                fj=fj_val,
                historical_exposure_status=exp_status,
                has_historical_exposure=has_hist,
                source_batch=str(entry.get("source_batch", "unknown")),
                source_core_sha256=source_core_sha,
                annual_frames=annual_frames,
                annual_weeks=annual_weeks,
                win_data_dir=str(entry.get("win_data_dir", "")),
                wsl_data_dir=str(entry.get("wsl_data_dir", "")),
                win_reader_entry=str(entry.get("win_reader_entry", "")),
                wsl_reader_entry=str(entry.get("wsl_reader_entry", "")),
            )

        if declared_partition_by_tile and set(declared_partition_by_tile.keys()) != set(self.tiles.keys()):
            missing = sorted(set(declared_partition_by_tile.keys()) ^ set(self.tiles.keys()))
            raise ValueError(
                f"Manifest `partitions` and `tiles` sets differ on {len(missing)} tiles: {missing[:5]}"
            )

        self._ordered_fit_tiles: Tuple[str, ...] = self._build_nested_fit_order()
        self._verify_zero_cross_split_overlap()

    @staticmethod
    def _normalize_tiles_collection(raw_tiles: Any) -> Dict[str, Mapping[str, Any]]:
        if isinstance(raw_tiles, Mapping):
            return {str(k): v for k, v in raw_tiles.items()}
        if isinstance(raw_tiles, list):
            out: Dict[str, Mapping[str, Any]] = {}
            for item in raw_tiles:
                tid = str(item["tile_id"])
                if tid in out:
                    raise RuntimeError(f"Duplicate tile_id {tid} in JSON list!")
                out[tid] = item
            return out
        return {}

    @staticmethod
    def _extract_authoritative_4way_role(info: Mapping[str, Any]) -> str:
        if "role" in info and info["role"] in ("fit", "calibration", "val", "test"):
            return str(info["role"])
        ds_role = str(info.get("dataset_role", ""))
        if ds_role == "train":
            tp = str(info.get("training_partition", "fit"))
            if tp not in ("fit", "calibration"):
                raise RuntimeError(f"Invalid training_partition {tp!r} for train tile {info.get('tile_id')}")
            return tp
        if ds_role in ("val", "test", "fit", "calibration"):
            return ds_role
        raise RuntimeError(f"Cannot resolve authoritative 4-way role from {info}")

    @staticmethod
    def _extract_manifest_4way_role(entry: Mapping[str, Any], fallback: str) -> str:
        if "authoritative_dataset_role" in entry:
            ds_role = str(entry["authoritative_dataset_role"])
            if ds_role == "train":
                tp = entry.get("authoritative_training_partition")
                if tp in ("fit", "calibration"):
                    return str(tp)
                c_role = str(entry.get("canonical_role", ""))
                if c_role == "calibration":
                    return "calibration"
                return "fit"
            if ds_role in ("val", "test", "fit", "calibration"):
                return ds_role

        c_role = str(entry.get("canonical_role", "")).strip()
        if c_role in ("fit", "student_holdout"):
            return "fit"
        if c_role == "calibration":
            return "calibration"
        if c_role == "val_spatial_clean" or c_role.startswith("val"):
            return "val"
        if c_role in ("test_spatial_unexposed", "test_spatial_legacy_reassigned") or c_role.startswith("test"):
            return "test"
        return fallback

    @staticmethod
    def _map_to_canonical_split(
        tile_id: str, authoritative_role: str, historical_exposure_status: str
    ) -> str:
        if tile_id in ADDITIONAL_FIT_HOLDOUT:
            return "student_holdout"
        if authoritative_role == "fit":
            return "fit"
        if authoritative_role == "calibration":
            return "calibration"
        if authoritative_role == "val":
            return "val_spatial_clean"
        if authoritative_role == "test":
            if historical_exposure_status == "unexposed":
                return "test_spatial_unexposed"
            if historical_exposure_status == "exposed":
                return "test_spatial_legacy_reassigned"
            raise ValueError(
                f"Test tile {tile_id} has non-resolvable exposure status {historical_exposure_status!r}; "
                f"strictly blocked from test_spatial_unexposed."
            )
        raise RuntimeError(f"Unknown authoritative_role {authoritative_role!r} for {tile_id}")

    def _build_nested_fit_order(self) -> Tuple[str, ...]:
        """Build deterministic spatially stratified order so fit_8 ⊂ fit_17 ⊂ fit_34 ⊂ fit_67 ⊆ fit_all."""
        by_group: Dict[str, List[str]] = {}
        for tid, rec in self.tiles.items():
            if rec.canonical_split == "fit":
                if tid in ADDITIONAL_FIT_HOLDOUT:
                    raise RuntimeError(f"Holdout tile {tid} leaked into fit candidates!")
                by_group.setdefault(rec.spatial_group, []).append(tid)

        ordered_groups = sorted(
            by_group.keys(),
            key=lambda g: _stable_rank_key(f"{BENCHMARK_CONTRACT_ID}::fit_group", g),
        )
        for g in ordered_groups:
            by_group[g].sort(
                key=lambda tid: _stable_rank_key(f"{BENCHMARK_CONTRACT_ID}::fit_tile", tid)
            )

        result: List[str] = []
        round_idx = 0
        while True:
            added = False
            for g in ordered_groups:
                bucket = by_group[g]
                if round_idx < len(bucket):
                    result.append(bucket[round_idx])
                    added = True
            if not added:
                break
            round_idx += 1
        return tuple(result)

    def _verify_zero_cross_split_overlap(self) -> None:
        buckets: Dict[str, set[str]] = {k: set() for k in CANONICAL_ALL_BUCKETS}
        spatial_groups_by_macro: Dict[str, set[str]] = {
            "fit": set(),
            "calibration": set(),
            "val": set(),
            "test": set(),
        }
        for tid, rec in self.tiles.items():
            buckets[rec.canonical_split].add(tid)
            if rec.canonical_split == "fit":
                spatial_groups_by_macro["fit"].add(rec.spatial_group)
            elif rec.canonical_split == "calibration":
                spatial_groups_by_macro["calibration"].add(rec.spatial_group)
            elif rec.canonical_split == "val_spatial_clean":
                spatial_groups_by_macro["val"].add(rec.spatial_group)
            elif rec.canonical_split in ("test_spatial_unexposed", "test_spatial_legacy_reassigned"):
                spatial_groups_by_macro["test"].add(rec.spatial_group)

        keys = list(buckets.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                inter = buckets[keys[i]] & buckets[keys[j]]
                if inter:
                    raise RuntimeError(
                        f"Tile overlap between {keys[i]} and {keys[j]}: {sorted(inter)}"
                    )

        macro_keys = list(spatial_groups_by_macro.keys())
        for i in range(len(macro_keys)):
            for j in range(i + 1, len(macro_keys)):
                sg_inter = (
                    spatial_groups_by_macro[macro_keys[i]]
                    & spatial_groups_by_macro[macro_keys[j]]
                )
                if sg_inter:
                    raise RuntimeError(
                        f"1km spatial_group overlap between {macro_keys[i]} and {macro_keys[j]}: "
                        f"{sorted(sg_inter)}"
                    )

    def get_fit_tiles(self, subset_name: str = "fit_all") -> List[str]:
        if subset_name not in FIT_SUBSET_SIZES:
            raise ValueError(
                f"Unknown fit subset {subset_name!r}. Valid subsets: {sorted(FIT_SUBSET_SIZES)}"
            )
        target_n = FIT_SUBSET_SIZES[subset_name]
        if target_n is None:
            selected = list(self._ordered_fit_tiles)
        else:
            if len(self._ordered_fit_tiles) < target_n:
                raise RuntimeError(
                    f"Requested {subset_name} ({target_n} tiles), but only "
                    f"{len(self._ordered_fit_tiles)} fit tiles are registered in manifest."
                )
            selected = list(self._ordered_fit_tiles[:target_n])
        self.assert_fit_allowed(selected)
        return selected

    def get_calibration_tiles(self) -> List[str]:
        tiles = sorted(
            tid for tid, rec in self.tiles.items() if rec.canonical_split == "calibration"
        )
        self.assert_calibration_allowed(tiles)
        return tiles

    def verify_blind_unseal_manifest(
        self,
        unseal_freeze_manifest: Mapping[str, Any],
        *,
        one_shot_only: bool = True,
    ) -> None:
        """Verify pre-registration freeze manifest before allowing `test_spatial_unexposed` access."""
        if not isinstance(unseal_freeze_manifest, Mapping):
            raise RuntimeError("unseal_freeze_manifest must be a JSON object / mapping.")
        if one_shot_only and self.blind_open_count > 0:
            raise RuntimeError(
                f"BLIND TEST LOCK VIOLATION: `test_spatial_unexposed` was already unsealed "
                f"(blind_open_count={self.blind_open_count}); repeated blind-test runs are forbidden!"
            )
        req_keys = (
            "benchmark_contract",
            "split_sha256",
            "manifest_sha256",
            "selection_rule",
            "frozen_checkpoints",
            "evaluator_version",
        )
        missing = [k for k in req_keys if not unseal_freeze_manifest.get(k)]
        if missing:
            raise RuntimeError(
                f"Invalid blind-test freeze manifest; missing required frozen fields: {missing}"
            )
        if unseal_freeze_manifest["benchmark_contract"] != BENCHMARK_CONTRACT_ID:
            raise RuntimeError(
                f"Freeze manifest benchmark_contract mismatch: "
                f"{unseal_freeze_manifest['benchmark_contract']} != {BENCHMARK_CONTRACT_ID}"
            )
        if unseal_freeze_manifest["split_sha256"] != self.split_sha256:
            raise RuntimeError(
                f"Freeze manifest split_sha256 ({unseal_freeze_manifest['split_sha256']}) != "
                f"active {self.split_sha256}"
            )
        if unseal_freeze_manifest["manifest_sha256"] != self.manifest_sha256:
            raise RuntimeError(
                f"Freeze manifest manifest_sha256 ({unseal_freeze_manifest['manifest_sha256']}) != "
                f"active {self.manifest_sha256}"
            )
        ckpts = unseal_freeze_manifest.get("frozen_checkpoints")
        if not isinstance(ckpts, list) or len(ckpts) == 0:
            raise RuntimeError("Freeze manifest `frozen_checkpoints` must be a non-empty list.")
        for item in ckpts:
            if not isinstance(item, Mapping) or not item.get("checkpoint_sha256"):
                raise RuntimeError(
                    "Every entry in `frozen_checkpoints` must include `checkpoint_sha256`."
                )

    def get_eval_tiles(
        self,
        bucket_name: str,
        *,
        model_provenance: Optional[Mapping[str, Any]] = None,
        allow_empty: bool = False,
        unseal_freeze_manifest: Optional[Mapping[str, Any]] = None,
        one_shot_blind_unseal: bool = True,
    ) -> List[str]:
        if bucket_name not in CANONICAL_EVAL_BUCKETS:
            raise ValueError(
                f"Unknown evaluation bucket {bucket_name!r}. "
                f"Canonical buckets (never merged): {CANONICAL_EVAL_BUCKETS}"
            )
        if bucket_name == "test_spatial_unexposed":
            if unseal_freeze_manifest is None:
                raise RuntimeError(
                    "BLIND TEST LOCK: `test_spatial_unexposed` cannot be accessed during "
                    "development (`run-dev-suite` / `eval`). Pass an explicit pre-registered "
                    "`unseal_freeze_manifest` via `unseal-blind-test` only ONCE after freezing "
                    "all checkpoints, calibration thresholds, and selection rules."
                )
            self.verify_blind_unseal_manifest(
                unseal_freeze_manifest, one_shot_only=one_shot_blind_unseal
            )

        tiles = sorted(
            tid for tid, rec in self.tiles.items() if rec.canonical_split == bucket_name
        )
        if not tiles and not allow_empty:
            raise RuntimeError(
                f"Evaluation bucket {bucket_name!r} currently has 0 ready tiles in {self.manifest_path.name}."
            )
        if tiles:
            self.assert_eval_bucket_allowed(
                bucket_name,
                tiles,
                model_provenance=model_provenance,
                unseal_freeze_manifest=unseal_freeze_manifest,
                check_one_shot=False,
            )
        if bucket_name == "test_spatial_unexposed":
            lock_path = self.blind_ledger_path.with_name(f".{self.blind_ledger_path.name}.lock")
            with exclusive_file_lock(lock_path):
                disk_entries = self._load_persistent_blind_ledger()
                current_count = max(int(self.blind_open_count), len(disk_entries))
                if one_shot_blind_unseal and current_count > 0:
                    self._blind_unseal_records = list(disk_entries)
                    self.blind_open_count = current_count
                    raise RuntimeError(
                        f"BLIND TEST LOCK VIOLATION: `test_spatial_unexposed` was already unsealed "
                        f"(blind_open_count={current_count}); repeated blind-test runs are forbidden!"
                    )
                self.blind_open_count = current_count + 1
                rec_entry = {
                    "open_index": self.blind_open_count,
                    "split_sha256": self.split_sha256,
                    "manifest_sha256": self.manifest_sha256,
                    "tile_count": len(tiles),
                    "tile_ids": list(tiles),
                    "selection_rule": str(unseal_freeze_manifest.get("selection_rule", "")),
                }
                self._append_persistent_blind_ledger(rec_entry)
                self._blind_unseal_records = [*disk_entries, rec_entry]
                self._session_blind_unsealed_tiles.update(tiles)
        return tiles

    def _load_persistent_blind_ledger(self) -> List[Dict[str, Any]]:
        if not self.blind_ledger_path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for raw_line in self.blind_ledger_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if (
                isinstance(item, dict)
                and item.get("split_sha256") == self.split_sha256
                and item.get("manifest_sha256") == self.manifest_sha256
            ):
                entries.append(item)
        return entries

    def _append_persistent_blind_ledger(self, rec_entry: Mapping[str, Any]) -> None:
        self.blind_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.blind_ledger_path.open("a", encoding="utf-8") as f_led:
            f_led.write(json.dumps(dict(rec_entry), sort_keys=True) + "\n")
            f_led.flush()
            os.fsync(f_led.fileno())

    @property
    def is_blind_unsealed(self) -> bool:
        return len(self._session_blind_unsealed_tiles) > 0

    @property
    def _blind_unsealed_in_session(self) -> bool:
        return self.is_blind_unsealed

    def assert_tile_readable(self, tile_id: str) -> TileSplitRecord:
        """Enforce blind-test lock on ALL raw & streaming tile readers (`load_tile_fullday`, `prepare_tile_streaming_context`)."""
        if tile_id not in self.tiles:
            raise KeyError(f"Tile {tile_id!r} not registered in authoritative SplitContract")
        rec = self.tiles[tile_id]
        if (
            rec.canonical_split == "test_spatial_unexposed"
            and tile_id not in self._session_blind_unsealed_tiles
        ):
            raise RuntimeError(
                f"BLIND TEST LOCK: refusing to read or stream sealed blind tile {tile_id!r} "
                f"(`test_spatial_unexposed`) without active `unseal_freeze_manifest` authorization."
            )
        return rec

    def assert_tile_in_bucket(self, tile_id: str, bucket_name: str) -> TileSplitRecord:
        """Verify that `tile_id` is registered and belongs to `bucket_name`."""
        if bucket_name not in CANONICAL_ALL_BUCKETS:
            raise ValueError(
                f"Unknown bucket {bucket_name!r}; must be one of {CANONICAL_ALL_BUCKETS}."
            )
        rec = self.tiles.get(tile_id)
        if rec is None:
            raise RuntimeError(f"Tile {tile_id!r} is not registered in {self.manifest_path.name}.")
        if rec.canonical_split != bucket_name:
            raise RuntimeError(
                f"Tile bucket mismatch for {tile_id!r}: requested bucket={bucket_name!r}, "
                f"actual canonical_split={rec.canonical_split!r}."
            )
        if bucket_name == "test_spatial_unexposed":
            raise RuntimeError(
                f"BLIND TEST LOCK: inverse-design or ad-hoc inspection on blind tile {tile_id!r} "
                f"(`test_spatial_unexposed`) is prohibited during development."
            )
        return rec

    def assert_fit_allowed(self, tile_ids: Sequence[str]) -> None:
        if not tile_ids:
            raise RuntimeError("Empty tile list passed to `assert_fit_allowed`.")
        violations: List[str] = []
        for tid in tile_ids:
            if tid in ADDITIONAL_FIT_HOLDOUT:
                violations.append(f"{tid}(student_holdout)")
                continue
            rec = self.tiles.get(tid)
            if rec is None:
                u_role = self.universe_tile_roles.get(tid, "unknown")
                if u_role != "fit":
                    violations.append(f"{tid}(universe_role={u_role})")
                else:
                    violations.append(f"{tid}(unregistered_in_manifest)")
                continue
            if rec.canonical_split != "fit" or rec.authoritative_role != "fit":
                violations.append(f"{tid}(canonical_split={rec.canonical_split})")
        if violations:
            raise RuntimeError(
                "RUNTIME FIT FIREWALL BLOCKED NON-FIT TILES FROM TRAINING/NORMALIZATION: "
                + ", ".join(violations)
            )

    def issue_fit_firewall_token(
        self, tile_ids: Sequence[str], subset_name: str = "fit_all"
    ) -> FitFirewallToken:
        self.assert_fit_allowed(tile_ids)
        frozen = frozenset(tile_ids)
        sig = _compute_firewall_signature(
            self.split_sha256, self.manifest_sha256, subset_name, frozen
        )
        return FitFirewallToken(
            benchmark_contract=BENCHMARK_CONTRACT_ID,
            split_sha256=self.split_sha256,
            manifest_sha256=self.manifest_sha256,
            subset_name=subset_name,
            allowed_tile_ids=frozen,
            _signature=sig,
        )

    def assert_calibration_allowed(self, tile_ids: Sequence[str]) -> None:
        if not tile_ids:
            raise RuntimeError("Empty tile list passed to `assert_calibration_allowed`.")
        violations: List[str] = []
        for tid in tile_ids:
            rec = self.tiles.get(tid)
            if rec is None:
                violations.append(f"{tid}(unregistered)")
            elif rec.canonical_split != "calibration" or rec.authoritative_role != "calibration":
                violations.append(f"{tid}(canonical_split={rec.canonical_split})")
        if violations:
            raise RuntimeError(
                "RUNTIME CALIBRATION FIREWALL BLOCKED NON-CALIBRATION TILES: "
                + ", ".join(violations)
            )

    def assert_eval_bucket_allowed(
        self,
        bucket_name: str,
        tile_ids: Sequence[str],
        *,
        model_provenance: Optional[Mapping[str, Any]] = None,
        require_calibrated_provenance: bool = False,
        unseal_freeze_manifest: Optional[Mapping[str, Any]] = None,
        check_one_shot: bool = True,
    ) -> None:
        if bucket_name not in CANONICAL_EVAL_BUCKETS:
            raise RuntimeError(
                f"Invalid evaluation bucket {bucket_name!r}; must be one of {CANONICAL_EVAL_BUCKETS}."
            )
        if not tile_ids:
            raise RuntimeError(f"Empty tile list passed to `assert_eval_bucket_allowed({bucket_name})`.")

        if bucket_name == "test_spatial_unexposed":
            if unseal_freeze_manifest is None:
                raise RuntimeError(
                    "BLIND TEST LOCK: `test_spatial_unexposed` evaluation requires `unseal_freeze_manifest`."
                )
            self.verify_blind_unseal_manifest(unseal_freeze_manifest, one_shot_only=check_one_shot)

        violations: List[str] = []
        for tid in tile_ids:
            rec = self.tiles.get(tid)
            if rec is None:
                violations.append(f"{tid}(unregistered)")
                continue
            if rec.canonical_split != bucket_name:
                violations.append(f"{tid}(actual={rec.canonical_split}, expected={bucket_name})")
                continue
            if bucket_name == "test_spatial_unexposed":
                if rec.historical_exposure_status != "unexposed" or rec.has_historical_exposure:
                    violations.append(
                        f"{tid}(historical_exposure_status={rec.historical_exposure_status})"
                    )
        if violations:
            raise RuntimeError(
                f"RUNTIME EVAL BUCKET FIREWALL VIOLATION for {bucket_name!r}: "
                + ", ".join(violations)
            )

        if require_calibrated_provenance and model_provenance is None:
            raise RuntimeError(
                f"Evaluation on {bucket_name!r} requires complete checkpoint `model_provenance`."
            )

        if model_provenance is not None:
            ckpt_split_sha = model_provenance.get("split_contract_sha256")
            if ckpt_split_sha != self.split_sha256:
                raise RuntimeError(
                    f"Checkpoint split_contract_sha256 ({ckpt_split_sha}) != "
                    f"authoritative {self.split_sha256}"
                )
            ckpt_manifest_sha = model_provenance.get("manifest_sha256")
            if ckpt_manifest_sha is not None and ckpt_manifest_sha != self.manifest_sha256:
                raise RuntimeError(
                    f"Checkpoint manifest_sha256 ({ckpt_manifest_sha}) != "
                    f"active {self.manifest_sha256}"
                )
            feat_ver = model_provenance.get("feature_version")
            if feat_ver is not None and feat_ver != FEATURE_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Checkpoint feature_version ({feat_ver}) != {FEATURE_SCHEMA_VERSION}"
                )

            raw_fit_tiles = list(model_provenance.get("fit_tile_ids") or [])
            if not raw_fit_tiles:
                raise RuntimeError(
                    "Checkpoint provenance has empty `fit_tile_ids`; refusing to evaluate unverified model!"
                )
            fit_tiles = set(raw_fit_tiles)
            overlap = fit_tiles & set(tile_ids)
            if overlap:
                raise RuntimeError(
                    f"Checkpoint fit_tile_ids overlap evaluation bucket {bucket_name}: {sorted(overlap)}"
                )
            self.assert_fit_allowed(sorted(fit_tiles))

            if require_calibrated_provenance or model_provenance.get("calibrated") is True:
                if model_provenance.get("calibrated") is not True:
                    raise RuntimeError(
                        "Checkpoint provenance indicates `calibrated != True`; run `calibrate` on "
                        "`calibration` split before formal evaluation."
                    )
                raw_cal_tiles = list(model_provenance.get("calibration_tile_ids") or [])
                if not raw_cal_tiles:
                    raise RuntimeError(
                        "Calibrated checkpoint provenance has empty `calibration_tile_ids`!"
                    )
                cal_tiles = set(raw_cal_tiles)
                if cal_tiles & set(tile_ids):
                    raise RuntimeError(
                        f"Checkpoint calibration_tile_ids overlap evaluation bucket {bucket_name}!"
                    )
                if cal_tiles & fit_tiles:
                    raise RuntimeError("Checkpoint calibration_tile_ids overlap fit_tile_ids!")
                self.assert_calibration_allowed(sorted(cal_tiles))

        if bucket_name in ("test_spatial_legacy_reassigned", "test_spatial_unexposed"):
            if model_provenance is None or model_provenance.get("trained_from_scratch_fit_only") is not True:
                raise RuntimeError(
                    f"Evaluation on `{bucket_name}` requires explicit checkpoint "
                    "provenance `trained_from_scratch_fit_only == True`."
                )

    def build_verification_report(self) -> Dict[str, Any]:
        counts_by_split: Dict[str, int] = {k: 0 for k in CANONICAL_ALL_BUCKETS}
        frames_by_split: Dict[str, int] = {k: 0 for k in CANONICAL_ALL_BUCKETS}
        exposure_by_split: Dict[str, Dict[str, int]] = {
            k: {"exposed": 0, "unexposed": 0, "unknown": 0} for k in CANONICAL_ALL_BUCKETS
        }
        spatial_groups_by_split: Dict[str, set[str]] = {k: set() for k in CANONICAL_ALL_BUCKETS}
        for rec in self.tiles.values():
            counts_by_split[rec.canonical_split] += 1
            frames_by_split[rec.canonical_split] += rec.annual_frames
            exposure_by_split[rec.canonical_split][rec.historical_exposure_status] += 1
            spatial_groups_by_split[rec.canonical_split].add(rec.spatial_group)

        nested_subsets: Dict[str, List[str]] = {}
        for name in ("fit_8", "fit_17", "fit_34", "fit_67", "fit_all"):
            size = FIT_SUBSET_SIZES[name]
            if size is None or size <= len(self._ordered_fit_tiles):
                nested_subsets[name] = self.get_fit_tiles(name)

        nested_chain_valid = True
        chain_keys = [k for k in ("fit_8", "fit_17", "fit_34", "fit_67", "fit_all") if k in nested_subsets]
        for a, b in zip(chain_keys[:-1], chain_keys[1:]):
            set_a, set_b = set(nested_subsets[a]), set(nested_subsets[b])
            if not (set_a <= set_b):
                nested_chain_valid = False

        val_tiles = [self.tiles[t] for t in self.get_eval_tiles("val_spatial_clean", allow_empty=True)]
        val_unexposed_n = sum(1 for r in val_tiles if r.historical_exposure_status == "unexposed")
        val_historical_n = sum(1 for r in val_tiles if r.historical_exposure_status == "exposed")

        csv_info: Dict[str, Any] = {
            "csv_found": False,
            "csv_path": None,
            "csv_sha256": None,
            "pinned_snapshot_v1_csv_sha256_matched": None,
            "manifest_csv_row_parity_verified": None,
            "discrepancies": [],
        }
        wm_root = discover_world_model_root(self.manifest_path.parent)
        if wm_root is not None and not self.is_synthetic_fixture:
            cand_csv = (wm_root / REL_ANNUAL_CATALOG_CSV).resolve()
            if cand_csv.exists():
                csv_sha = compute_file_sha256(cand_csv)
                discrepancies: List[str] = []
                seen_csv_tiles: set[str] = set()
                with cand_csv.open("r", encoding="utf-8-sig", newline="") as f_csv:
                    reader = csv.DictReader(f_csv)
                    for row in reader:
                        tid = str(row.get("地块", "")).strip()
                        if not tid:
                            continue
                        seen_csv_tiles.add(tid)
                        rec = self.tiles.get(tid)
                        if rec is None:
                            discrepancies.append(f"csv_only:{tid}")
                            continue
                        csv_role = str(row.get("标准分区角色_canonical_role", "")).strip()
                        if csv_role and csv_role != rec.canonical_split:
                            discrepancies.append(f"role_mismatch:{tid}:{csv_role}!={rec.canonical_split}")
                        csv_exp = str(row.get("历史曝光三态_historical_exposure_status", "")).strip()
                        if csv_exp and csv_exp != rec.historical_exposure_status:
                            discrepancies.append(
                                f"exposure_mismatch:{tid}:{csv_exp}!={rec.historical_exposure_status}"
                            )
                        csv_core_sha = str(row.get("源核心SHA256", "")).strip().lower()
                        if csv_core_sha and rec.source_core_sha256 and csv_core_sha != rec.source_core_sha256:
                            discrepancies.append(f"core_sha_mismatch:{tid}")
                for tid in sorted(set(self.tiles.keys()) - seen_csv_tiles):
                    discrepancies.append(f"manifest_only:{tid}")
                csv_info = {
                    "csv_found": True,
                    "csv_path": str(cand_csv),
                    "csv_sha256": csv_sha,
                    "pinned_snapshot_v1_csv_sha256_matched": csv_sha == EXPECTED_SNAPSHOT_V1_CSV_SHA256,
                    "manifest_csv_row_parity_verified": len(discrepancies) == 0,
                    "discrepancies": discrepancies,
                }

        return {
            "benchmark_contract": BENCHMARK_CONTRACT_ID,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "fixture_kind": self.fixture_kind,
            "is_synthetic_fixture": self.is_synthetic_fixture,
            "split_path": str(self.split_path),
            "split_sha256": self.split_sha256,
            "pinned_split_sha256_matched": self.split_sha256 == EXPECTED_MANHATTAN_4445_SPLIT_SHA256,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "pinned_snapshot_v1_manifest_sha256_matched": (
                self.manifest_sha256 == EXPECTED_SNAPSHOT_V1_MANIFEST_SHA256
            ),
            "catalog_csv_verification": csv_info,
            "universe_tile_count": len(self.universe_tile_roles),
            "ready_annual_tile_count": len(self.tiles),
            "additional_fit_holdout": sorted(ADDITIONAL_FIT_HOLDOUT),
            "counts_by_canonical_split": counts_by_split,
            "annual_frames_by_canonical_split": frames_by_split,
            "spatial_group_counts_by_split": {
                k: len(v) for k, v in spatial_groups_by_split.items()
            },
            "exposure_tri_state_by_split": exposure_by_split,
            "val_spatial_clean_provenance": {
                "total_tiles": len(val_tiles),
                "unexposed_tiles": val_unexposed_n,
                "historical_provenance_tiles": val_historical_n,
            },
            "blind_test_lock_status": {
                "bucket": "test_spatial_unexposed",
                "registered_tiles": counts_by_split["test_spatial_unexposed"],
                "blind_open_count": self.blind_open_count,
                "blind_ledger_path": str(self.blind_ledger_path),
                "blind_ledger_exists": self.blind_ledger_path.exists(),
                "historical_blind_unseal_count": len(self._blind_unseal_records),
                "session_blind_unsealed": self.is_blind_unsealed,
                "locked_for_development": self.blind_open_count == 0 and (not self.is_blind_unsealed),
            },
            "nested_fit_subset_counts": {k: len(v) for k, v in nested_subsets.items()},
            "nested_fit_chain_verified": nested_chain_valid,
            "spatial_1km_block_disjoint_verified": True,
            "cross_split_tile_overlap_count": 0,
        }
