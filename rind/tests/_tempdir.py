"""Sandbox-resilient temporary directory helper for `rind.tests`."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import uuid


class TemporaryTestDir:
    """Context manager using system temp dir when writable, else `<pkg_root>/.tmp_unittest/<uuid>`."""

    def __init__(self, prefix: str = "rind_test_") -> None:
        self.prefix = prefix
        self._std_tmp: tempfile.TemporaryDirectory[str] | None = None
        self._fallback_path: Path | None = None

    def __enter__(self) -> str:
        try:
            td = tempfile.TemporaryDirectory(prefix=self.prefix)
            p = Path(td.name)
            probe = p / ".probe"
            probe.write_bytes(b"ok")
            probe.unlink()
            self._std_tmp = td
            return str(p)
        except Exception:
            if self._std_tmp is not None:
                try:
                    self._std_tmp.cleanup()
                except Exception:
                    pass
                self._std_tmp = None
            pkg_root = Path(__file__).resolve().parents[2]
            base = pkg_root / ".tmp_unittest"
            base.mkdir(parents=True, exist_ok=True)
            sub = base / f"{self.prefix}{uuid.uuid4().hex[:10]}"
            sub.mkdir(parents=True, exist_ok=False)
            self._fallback_path = sub
            return str(sub)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._std_tmp is not None:
            self._std_tmp.cleanup()
            self._std_tmp = None
        if self._fallback_path is not None:
            shutil.rmtree(self._fallback_path, ignore_errors=True)
            parent = self._fallback_path.parent
            try:
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            except Exception:
                pass
            self._fallback_path = None
