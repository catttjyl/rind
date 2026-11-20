"""Host & GPU Resource Guard (`MANHATTAN_4445_ANNUAL_18808_V1`, Fixes Astra P0-8).

Queries REAL host physical memory, commit/pagefile limits, process working set, and CUDA VRAM:
- Native Windows (`os.name == "nt"`): Win32 `GlobalMemoryStatusEx` + `GetProcessMemoryInfo` via `ctypes`.
  Fails closed (`RuntimeError`) if Win32 memory APIs fail.
- WSL (`"microsoft"` in kernel release): queries the real Windows host via `powershell.exe`
  (`Win32_OperatingSystem` & `Win32_PerfFormattedData_PerfOS_Memory`), NEVER trusting WSL `/proc/meminfo`
  alone. Fails closed if the Windows host telemetry query fails.
- macOS / native Linux: `sysctl hw.memsize` + `vm_stat` / `/proc/meminfo` + `resource`.
- CUDA: `torch.cuda.mem_get_info`, `memory_allocated`, `max_memory_allocated`, `memory_reserved`,
  `max_memory_reserved`, plus optional `nvidia-smi` external process inspection.
"""

from __future__ import annotations

import ctypes
import dataclasses
import gc
import json
import os
import platform
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


HARD_MAX_PHYSICAL_RAM_PERCENT = 95.0
DEFAULT_WARN_PHYSICAL_RAM_PERCENT = 92.0
DEFAULT_MIN_COMMIT_HEADROOM_BYTES = 1024 * 1024 * 1024  # 1 GiB
DEFAULT_MIN_VRAM_HEADROOM_BYTES = 512 * 1024 * 1024  # 512 MiB


def is_wsl_environment() -> bool:
    if os.name == "nt":
        return False
    rel = platform.uname().release.lower()
    return "microsoft" in rel or "wsl" in rel


@dataclasses.dataclass(frozen=True)
class HostResourceSnapshot:
    timestamp_unix_s: float
    stage: str
    host_os_kind: str  # "windows_native" | "wsl_windows_host" | "darwin" | "linux_native"
    physical_total_bytes: int
    physical_avail_bytes: int
    physical_used_bytes: int
    physical_used_percent: float
    commit_limit_bytes: int
    commit_avail_bytes: int
    commit_used_bytes: int
    commit_used_percent: float
    process_working_set_bytes: int
    process_commit_bytes: int
    cuda_available: bool
    cuda_device_name: str
    cuda_total_bytes: int
    cuda_free_bytes: int
    cuda_allocated_bytes: int
    cuda_max_allocated_bytes: int
    cuda_reserved_bytes: int
    cuda_max_reserved_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        gib = float(1024 ** 3)
        mib = float(1024 ** 2)
        return {
            "timestamp_unix_s": round(self.timestamp_unix_s, 3),
            "stage": self.stage,
            "host_os_kind": self.host_os_kind,
            "physical_total_gib": round(self.physical_total_bytes / gib, 4),
            "physical_avail_gib": round(self.physical_avail_bytes / gib, 4),
            "physical_used_gib": round(self.physical_used_bytes / gib, 4),
            "physical_used_percent": round(self.physical_used_percent, 3),
            "commit_limit_gib": round(self.commit_limit_bytes / gib, 4),
            "commit_avail_gib": round(self.commit_avail_bytes / gib, 4),
            "commit_used_gib": round(self.commit_used_bytes / gib, 4),
            "commit_used_percent": round(self.commit_used_percent, 3),
            "process_working_set_mib": round(self.process_working_set_bytes / mib, 3),
            "process_commit_mib": round(self.process_commit_bytes / mib, 3),
            "cuda_available": self.cuda_available,
            "cuda_device_name": self.cuda_device_name,
            "cuda_total_mib": round(self.cuda_total_bytes / mib, 3),
            "cuda_free_mib": round(self.cuda_free_bytes / mib, 3),
            "cuda_allocated_mib": round(self.cuda_allocated_bytes / mib, 3),
            "cuda_max_allocated_mib": round(self.cuda_max_allocated_bytes / mib, 3),
            "cuda_reserved_mib": round(self.cuda_reserved_bytes / mib, 3),
            "cuda_max_reserved_mib": round(self.cuda_max_reserved_bytes / mib, 3),
        }


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _query_windows_native_memory() -> Dict[str, int | float | str]:
    kernel32 = getattr(ctypes, "windll", None)
    if kernel32 is None or not hasattr(kernel32, "kernel32"):
        raise RuntimeError("Windows ctypes.windll.kernel32 is unavailable; refusing to run blind.")

    mem = _MEMORYSTATUSEX()
    mem.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    if not ok or mem.ullTotalPhys <= 0:
        raise RuntimeError("Win32 GlobalMemoryStatusEx failed; failing closed per resource guard.")

    total_phys = int(mem.ullTotalPhys)
    avail_phys = int(mem.ullAvailPhys)
    used_phys = max(0, total_phys - avail_phys)
    phys_pct = 100.0 * float(used_phys) / float(total_phys)

    commit_limit = int(mem.ullTotalPageFile)
    commit_avail = int(mem.ullAvailPageFile)
    commit_used = max(0, commit_limit - commit_avail)
    commit_pct = (
        100.0 * float(commit_used) / float(commit_limit) if commit_limit > 0 else phys_pct
    )

    proc_ws = 0
    proc_commit = 0
    pmc = _PROCESS_MEMORY_COUNTERS_EX()
    pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS_EX)
    h_proc = ctypes.windll.kernel32.GetCurrentProcess()
    # K32GetProcessMemoryInfo in kernel32 or GetProcessMemoryInfo in psapi
    fn = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None)
    if fn is None and hasattr(ctypes.windll, "psapi"):
        fn = getattr(ctypes.windll.psapi, "GetProcessMemoryInfo", None)
    if fn is not None:
        if fn(h_proc, ctypes.byref(pmc), pmc.cb):
            proc_ws = int(pmc.WorkingSetSize)
            proc_commit = int(pmc.PrivateUsage or pmc.PagefileUsage)

    return {
        "host_os_kind": "windows_native",
        "physical_total_bytes": total_phys,
        "physical_avail_bytes": avail_phys,
        "physical_used_bytes": used_phys,
        "physical_used_percent": phys_pct,
        "commit_limit_bytes": commit_limit,
        "commit_avail_bytes": commit_avail,
        "commit_used_bytes": commit_used,
        "commit_used_percent": commit_pct,
        "process_working_set_bytes": proc_ws,
        "process_commit_bytes": proc_commit,
    }


def _query_wsl_windows_host_memory() -> Dict[str, int | float | str]:
    """Query the REAL Windows host from WSL; never rely on WSL /proc/meminfo alone."""
    ps_candidates = [
        "powershell.exe",
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
    ]
    ps_bin: Optional[str] = None
    for cand in ps_candidates:
        try:
            res = subprocess.run(
                [cand, "-NoProfile", "-Command", "$PSVersionTable.PSVersion.Major"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if res.returncode == 0:
                ps_bin = cand
                break
        except OSError:
            continue
    if ps_bin is None:
        raise RuntimeError(
            "WSL environment detected, but Windows host `powershell.exe` is unreachable. "
            "Refusing to trust WSL `/proc/meminfo` alone (fail-closed per P0-8)."
        )

    ps_script = (
        "$os = Get-CimInstance Win32_OperatingSystem; "
        "$pf = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; "
        "@{ "
        "TotalVisibleMemorySizeKB = [int64]$os.TotalVisibleMemorySize; "
        "FreePhysicalMemoryKB = [int64]$os.FreePhysicalMemory; "
        "TotalVirtualMemorySizeKB = [int64]$os.TotalVirtualMemorySize; "
        "FreeVirtualMemoryKB = [int64]$os.FreeVirtualMemory; "
        "CommittedBytes = [int64]$pf.CommittedBytes; "
        "CommitLimit = [int64]$pf.CommitLimit "
        "} | ConvertTo-Json -Compress"
    )
    res = subprocess.run(
        [ps_bin, "-NoProfile", "-Command", ps_script],
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError(
            f"WSL Windows host memory telemetry failed (rc={res.returncode}): {res.stderr.strip()}"
        )
    payload = json.loads(res.stdout.strip())
    total_phys = int(payload["TotalVisibleMemorySizeKB"]) * 1024
    avail_phys = int(payload["FreePhysicalMemoryKB"]) * 1024
    if total_phys <= 0:
        raise RuntimeError(f"Invalid Windows host TotalVisibleMemorySizeKB: {payload}")
    used_phys = max(0, total_phys - avail_phys)
    phys_pct = 100.0 * float(used_phys) / float(total_phys)

    commit_limit = int(payload.get("CommitLimit") or 0) or (
        int(payload["TotalVirtualMemorySizeKB"]) * 1024
    )
    commit_used = int(payload.get("CommittedBytes") or 0) or max(
        0, commit_limit - int(payload["FreeVirtualMemoryKB"]) * 1024
    )
    commit_avail = max(0, commit_limit - commit_used)
    commit_pct = (
        100.0 * float(commit_used) / float(commit_limit) if commit_limit > 0 else phys_pct
    )

    proc_rss = _query_posix_process_rss_bytes()
    return {
        "host_os_kind": "wsl_windows_host",
        "physical_total_bytes": total_phys,
        "physical_avail_bytes": avail_phys,
        "physical_used_bytes": used_phys,
        "physical_used_percent": phys_pct,
        "commit_limit_bytes": commit_limit,
        "commit_avail_bytes": commit_avail,
        "commit_used_bytes": commit_used,
        "commit_used_percent": commit_pct,
        "process_working_set_bytes": proc_rss,
        "process_commit_bytes": proc_rss,
    }


def _query_posix_process_rss_bytes() -> int:
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return int(ru)
        return int(ru) * 1024
    except Exception:
        return 0


def _query_darwin_memory() -> Dict[str, int | float | str]:
    try:
        total_out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        total_phys = int(total_out)
    except (OSError, subprocess.SubprocessError, ValueError):
        # Some macOS sandboxes deny hw.memsize while allowing the read-only
        # memory_pressure report. Keep real telemetry and fail closed if both fail.
        pressure = subprocess.run(
            ["memory_pressure"], capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        import re
        match = re.search(r"The system has (\d+)\s+\(", pressure)
        if match is None or int(match.group(1)) <= 0:
            raise RuntimeError("Cannot determine macOS physical memory from memory_pressure")
        total_phys = int(match.group(1))
    vm_out = subprocess.run(
        ["vm_stat"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout
    page_size = 4096
    pages: Dict[str, int] = {}
    for line in vm_out.splitlines():
        if "page size of" in line:
            parts = line.split("page size of")[-1].strip().split()
            if parts and parts[0].isdigit():
                page_size = int(parts[0])
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        val_str = v.strip().rstrip(".")
        if val_str.isdigit():
            pages[k.strip().lower()] = int(val_str)

    free_pages = (
        pages.get("pages free", 0)
        + pages.get("pages speculative", 0)
        + pages.get("pages inactive", 0)
        + pages.get("pages purgeable", 0)
    )
    avail_phys = min(total_phys, free_pages * page_size)
    used_phys = max(0, total_phys - avail_phys)
    phys_pct = 100.0 * float(used_phys) / float(total_phys)
    proc_rss = _query_posix_process_rss_bytes()
    return {
        "host_os_kind": "darwin",
        "physical_total_bytes": total_phys,
        "physical_avail_bytes": avail_phys,
        "physical_used_bytes": used_phys,
        "physical_used_percent": phys_pct,
        "commit_limit_bytes": total_phys,
        "commit_avail_bytes": avail_phys,
        "commit_used_bytes": used_phys,
        "commit_used_percent": phys_pct,
        "process_working_set_bytes": proc_rss,
        "process_commit_bytes": proc_rss,
    }


def _query_linux_native_memory() -> Dict[str, int | float | str]:
    info: Dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            k, rest = line.split(":", 1)
            toks = rest.strip().split()
            if toks and toks[0].isdigit():
                info[k.strip()] = int(toks[0]) * 1024
    total_phys = info.get("MemTotal", 0)
    if total_phys <= 0:
        raise RuntimeError("Linux /proc/meminfo returned MemTotal <= 0; failing closed.")
    avail_phys = info.get("MemAvailable", info.get("MemFree", 0))
    used_phys = max(0, total_phys - avail_phys)
    phys_pct = 100.0 * float(used_phys) / float(total_phys)

    commit_limit = info.get("CommitLimit", total_phys + info.get("SwapTotal", 0))
    commit_used = info.get("Committed_AS", used_phys)
    commit_avail = max(0, commit_limit - commit_used)
    commit_pct = (
        100.0 * float(commit_used) / float(commit_limit) if commit_limit > 0 else phys_pct
    )
    proc_rss = _query_posix_process_rss_bytes()
    return {
        "host_os_kind": "linux_native",
        "physical_total_bytes": total_phys,
        "physical_avail_bytes": avail_phys,
        "physical_used_bytes": used_phys,
        "physical_used_percent": phys_pct,
        "commit_limit_bytes": commit_limit,
        "commit_avail_bytes": commit_avail,
        "commit_used_bytes": commit_used,
        "commit_used_percent": commit_pct,
        "process_working_set_bytes": proc_rss,
        "process_commit_bytes": proc_rss,
    }


def sample_host_resources(
    stage: str = "unspecified",
    *,
    device: Optional[Any] = None,
    telemetry_override: Optional[Callable[[], Dict[str, int | float | str]]] = None,
) -> HostResourceSnapshot:
    """Collect an immediate snapshot of host RAM, commit/pagefile, and CUDA VRAM."""
    if telemetry_override is not None:
        host = telemetry_override()
    elif os.name == "nt":
        host = _query_windows_native_memory()
    elif is_wsl_environment():
        host = _query_wsl_windows_host_memory()
    elif platform.system() == "Darwin":
        host = _query_darwin_memory()
    else:
        host = _query_linux_native_memory()

    cuda_avail = bool(torch is not None and torch.cuda.is_available())
    dev_name = "cpu"
    c_total = 0
    c_free = 0
    c_alloc = 0
    c_max_alloc = 0
    c_res = 0
    c_max_res = 0
    if cuda_avail and torch is not None:
        dev_idx = 0
        if device is not None:
            dev_obj = torch.device(device)
            if dev_obj.type == "cuda":
                dev_idx = 0 if dev_obj.index is None else int(dev_obj.index)
        dev_name = str(torch.cuda.get_device_name(dev_idx))
        c_free, c_total = (int(x) for x in torch.cuda.mem_get_info(dev_idx))
        c_alloc = int(torch.cuda.memory_allocated(dev_idx))
        c_max_alloc = int(torch.cuda.max_memory_allocated(dev_idx))
        c_res = int(torch.cuda.memory_reserved(dev_idx))
        c_max_res = int(torch.cuda.max_memory_reserved(dev_idx))

    return HostResourceSnapshot(
        timestamp_unix_s=time.time(),
        stage=stage,
        host_os_kind=str(host["host_os_kind"]),
        physical_total_bytes=int(host["physical_total_bytes"]),
        physical_avail_bytes=int(host["physical_avail_bytes"]),
        physical_used_bytes=int(host["physical_used_bytes"]),
        physical_used_percent=float(host["physical_used_percent"]),
        commit_limit_bytes=int(host["commit_limit_bytes"]),
        commit_avail_bytes=int(host["commit_avail_bytes"]),
        commit_used_bytes=int(host["commit_used_bytes"]),
        commit_used_percent=float(host["commit_used_percent"]),
        process_working_set_bytes=int(host["process_working_set_bytes"]),
        process_commit_bytes=int(host["process_commit_bytes"]),
        cuda_available=cuda_avail,
        cuda_device_name=dev_name,
        cuda_total_bytes=c_total,
        cuda_free_bytes=c_free,
        cuda_allocated_bytes=c_alloc,
        cuda_max_allocated_bytes=c_max_alloc,
        cuda_reserved_bytes=c_res,
        cuda_max_reserved_bytes=c_max_res,
    )


class ResourceBudgetExceededError(RuntimeError):
    """Raised when host physical RAM, commit headroom, or CUDA VRAM headroom violates safety thresholds."""


class HostResourceGuard:
    """Active resource monitor enforcing < 95% host physical RAM and commit/VRAM headroom."""

    def __init__(
        self,
        *,
        max_physical_ram_percent: float = DEFAULT_WARN_PHYSICAL_RAM_PERCENT,
        hard_abort_ram_percent: float = HARD_MAX_PHYSICAL_RAM_PERCENT,
        min_commit_headroom_bytes: int = DEFAULT_MIN_COMMIT_HEADROOM_BYTES,
        min_vram_headroom_bytes: int = DEFAULT_MIN_VRAM_HEADROOM_BYTES,
        device: Optional[Any] = None,
        telemetry_override: Optional[Callable[[], Dict[str, int | float | str]]] = None,
    ) -> None:
        if hard_abort_ram_percent > HARD_MAX_PHYSICAL_RAM_PERCENT:
            raise ValueError(
                f"hard_abort_ram_percent ({hard_abort_ram_percent}) may not exceed {HARD_MAX_PHYSICAL_RAM_PERCENT}%"
            )
        self.max_physical_ram_percent = float(max_physical_ram_percent)
        self.hard_abort_ram_percent = float(hard_abort_ram_percent)
        self.min_commit_headroom_bytes = int(min_commit_headroom_bytes)
        self.min_vram_headroom_bytes = int(min_vram_headroom_bytes)
        self.device = device
        self._telemetry_override = telemetry_override
        self.snapshots: List[HostResourceSnapshot] = []
        self.cleanup_trigger_count: int = 0

    def _check_violations(self, snap: HostResourceSnapshot, threshold_pct: float) -> List[str]:
        reasons: List[str] = []
        if snap.physical_used_percent >= threshold_pct:
            reasons.append(
                f"physical_used_percent={snap.physical_used_percent:.2f}% >= {threshold_pct:.2f}%"
            )
        if snap.commit_limit_bytes > 0 and snap.commit_avail_bytes < self.min_commit_headroom_bytes:
            reasons.append(
                f"commit_avail={snap.commit_avail_bytes / (1024**3):.3f} GiB < "
                f"{self.min_commit_headroom_bytes / (1024**3):.3f} GiB"
            )
        if (
            snap.cuda_available
            and self.device is not None
            and str(self.device).startswith("cuda")
            and snap.cuda_total_bytes > 0
            and snap.cuda_free_bytes < self.min_vram_headroom_bytes
        ):
            reasons.append(
                f"cuda_free={snap.cuda_free_bytes / (1024**2):.1f} MiB < "
                f"{self.min_vram_headroom_bytes / (1024**2):.1f} MiB"
            )
        return reasons

    def check(
        self,
        stage: str,
        *,
        cleanup_callbacks: Optional[Sequence[Callable[[], None]]] = None,
    ) -> HostResourceSnapshot:
        snap = sample_host_resources(
            stage=stage,
            device=self.device,
            telemetry_override=self._telemetry_override,
        )
        self.snapshots.append(snap)

        soft_violations = self._check_violations(snap, self.max_physical_ram_percent)
        if soft_violations:
            self.cleanup_trigger_count += 1
            if cleanup_callbacks:
                for cb in cleanup_callbacks:
                    cb()
            gc.collect()
            if snap.cuda_available and torch is not None:
                torch.cuda.empty_cache()

            snap = sample_host_resources(
                stage=f"{stage}:post_cleanup",
                device=self.device,
                telemetry_override=self._telemetry_override,
            )
            self.snapshots.append(snap)

        hard_violations = self._check_violations(snap, self.hard_abort_ram_percent)
        if hard_violations:
            raise ResourceBudgetExceededError(
                f"HOST RESOURCE GUARD ABORT at stage={stage!r} ({snap.host_os_kind}): "
                + "; ".join(hard_violations)
            )
        return snap

    def summary(self) -> Dict[str, Any]:
        if not self.snapshots:
            self.check("summary_final")
        peak_phys_pct = max(s.physical_used_percent for s in self.snapshots)
        peak_commit_pct = max(s.commit_used_percent for s in self.snapshots)
        min_commit_avail = min(s.commit_avail_bytes for s in self.snapshots)
        peak_proc_ws = max(s.process_working_set_bytes for s in self.snapshots)
        peak_proc_commit = max(s.process_commit_bytes for s in self.snapshots)
        peak_cuda_alloc = max(s.cuda_max_allocated_bytes for s in self.snapshots)
        peak_cuda_res = max(s.cuda_max_reserved_bytes for s in self.snapshots)
        min_cuda_free = min(
            (s.cuda_free_bytes for s in self.snapshots if s.cuda_total_bytes > 0),
            default=0,
        )
        gib = float(1024 ** 3)
        mib = float(1024 ** 2)
        return {
            "host_os_kind": self.snapshots[-1].host_os_kind,
            "cuda_device_name": self.snapshots[-1].cuda_device_name,
            "sample_count": len(self.snapshots),
            "cleanup_trigger_count": self.cleanup_trigger_count,
            "hard_abort_ram_percent": self.hard_abort_ram_percent,
            "peak_physical_used_percent": round(peak_phys_pct, 3),
            "physical_ram_strictly_below_95pct": bool(peak_phys_pct < HARD_MAX_PHYSICAL_RAM_PERCENT),
            "peak_commit_used_percent": round(peak_commit_pct, 3),
            "min_commit_avail_gib": round(min_commit_avail / gib, 4),
            "peak_process_working_set_mib": round(peak_proc_ws / mib, 3),
            "peak_process_commit_mib": round(peak_proc_commit / mib, 3),
            "peak_cuda_allocated_mib": round(peak_cuda_alloc / mib, 3),
            "peak_cuda_reserved_mib": round(peak_cuda_res / mib, 3),
            "min_cuda_free_mib": round(min_cuda_free / mib, 3),
            "latest_snapshot": self.snapshots[-1].to_dict(),
        }
