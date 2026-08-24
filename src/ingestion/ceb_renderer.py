"""CEB page renderer backed by the installed Apabi ActiveX component.

The knowledge-base worker is a Python process running under WSL/Linux, while
the available Apabi Reader component is a 32-bit Windows ActiveX control.  We
therefore keep the Windows dependency behind a short-lived PowerShell child
process.  The rest of ingestion only sees a deterministic list of page PNGs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CEBRenderResult:
    """Rendered CEB pages and their stable derived-asset directory."""

    file_hash: str
    source_path: str
    output_dir: Path
    page_paths: List[Path]
    width: int
    height: int

    @property
    def page_count(self) -> int:
        return len(self.page_paths)


class CEBRenderer:
    """Render CEB pages to PNG without converting through PDF."""

    MANIFEST_NAME = "manifest.json"

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        cfg = self.config.get("ceb", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.width = max(320, int(cfg.get("render_width", 800)))
        self.height = max(480, int(cfg.get("render_height", 1000)))
        self.timeout = max(30, int(cfg.get("timeout", 300)))
        self.script = self._resolve_project_path(
            cfg.get("renderer_script", "scripts/ceb_render_pages.ps1")
        )
        self.powershell = str(cfg.get("powershell", "powershell.exe"))
        self.powershell32 = str(
            cfg.get(
                "powershell32",
                r"C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe",
            )
        )
        self.apabi_dir = str(cfg.get("apabi_dir", r"D:\Apabi reader"))
        paths = self.config.get("paths", {}) or {}
        render_root = paths.get("ceb_rendered")
        if not render_root:
            render_root = str(Path(paths.get("parsed_cache", "data/parsed_cache")) / "ceb_pages")
        self.render_root = Path(render_root)

    @staticmethod
    def _resolve_project_path(value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        # src/ingestion/ceb_renderer.py -> project root is parents[2].
        return Path(__file__).resolve().parents[2] / path

    @staticmethod
    def _windows_path(path: Path | str) -> str:
        raw = str(path)
        # The backend may also be run directly on Windows.
        if len(raw) >= 2 and raw[1] == ":":
            return raw.replace("/", "\\")
        try:
            converted = subprocess.check_output(
                ["wslpath", "-w", raw],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=10,
            ).strip()
            if converted:
                return converted
        except (OSError, subprocess.SubprocessError):
            pass
        return raw

    @staticmethod
    def _ps_quote(value: str) -> str:
        """Quote a value for a PowerShell single-quoted argument."""
        return "'" + str(value).replace("'", "''") + "'"

    def _manifest_path(self, output_dir: Path) -> Path:
        return output_dir / self.MANIFEST_NAME

    def _load_valid_cache(self, output_dir: Path, file_hash: str,
                          source_path: Path) -> Optional[CEBRenderResult]:
        manifest_path = self._manifest_path(output_dir)
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("file_hash") != file_hash
                or manifest.get("source_name") != source_path.name
                or int(manifest.get("width", 0)) != self.width
                or int(manifest.get("height", 0)) != self.height
            ):
                return None
            page_count = int(manifest.get("page_count", 0))
            page_paths = [
                output_dir / f"{page:04d}.png"
                for page in range(1, page_count + 1)
            ]
            if page_count <= 0 or any(
                not path.is_file() or path.stat().st_size <= 0 for path in page_paths
            ):
                return None
            return CEBRenderResult(
                file_hash=file_hash,
                source_path=str(source_path),
                output_dir=output_dir,
                page_paths=page_paths,
                width=self.width,
                height=self.height,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def render(self, source_path: str, file_hash: str,
               force: bool = False) -> CEBRenderResult:
        """Render every CEB page and return cached PNG paths."""
        if not self.enabled:
            raise RuntimeError("CEB 渲染器未启用")
        source = Path(source_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"CEB 文件不存在: {source}")
        if source.suffix.lower() != ".ceb":
            raise ValueError(f"CEB 渲染器只接受 .ceb 文件: {source}")
        if not self.script.is_file():
            raise RuntimeError(f"CEB 渲染脚本不存在: {self.script}")

        self.render_root.mkdir(parents=True, exist_ok=True)
        output_dir = self.render_root / file_hash
        if not force:
            cached = self._load_valid_cache(output_dir, file_hash, source)
            if cached:
                return cached

        stage_dir = self.render_root / f".{file_hash}.rendering-{os.getpid()}"
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True, exist_ok=False)

        script_win = self._windows_path(self.script)
        input_win = self._windows_path(source)
        output_win = self._windows_path(stage_dir)
        nested = " ".join([
            self._ps_quote(self.powershell32),
            "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File",
            self._ps_quote(script_win),
            "-InputPath", self._ps_quote(input_win),
            "-OutputDir", self._ps_quote(output_win),
            "-ApabiDir", self._ps_quote(self._windows_path(self.apabi_dir)),
            "-Width", str(self.width),
            "-Height", str(self.height),
        ])
        command = [
            self.powershell,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "& " + nested,
        ]

        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(
                    f"CEB 分页渲染失败 (exit={completed.returncode}): {detail[-1200:]}"
                )

            page_paths = sorted(stage_dir.glob("[0-9][0-9][0-9][0-9].png"))
            if not page_paths:
                detail = (completed.stdout or completed.stderr or "").strip()
                raise RuntimeError(f"CEB 渲染未生成 PNG: {detail[-800:]}")

            manifest = {
                "file_hash": file_hash,
                "source_name": source.name,
                "source_size": source.stat().st_size,
                "page_count": len(page_paths),
                "width": self.width,
                "height": self.height,
                "renderer": "apabi-cebviewer-activex",
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            }
            self._manifest_path(stage_dir).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
            stage_dir.rename(output_dir)
            return CEBRenderResult(
                file_hash=file_hash,
                source_path=str(source),
                output_dir=output_dir,
                page_paths=[output_dir / path.name for path in page_paths],
                width=self.width,
                height=self.height,
            )
        except Exception:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise

    def cleanup(self, file_hash: str) -> None:
        """Remove derived CEB page images for a deleted file."""
        shutil.rmtree(self.render_root / file_hash, ignore_errors=True)
