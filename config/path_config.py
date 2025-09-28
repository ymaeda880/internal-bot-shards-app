# config/path_config.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import os
import streamlit as st

AVAILABLE_PRESETS = ("Home", "Portable", "PrecMacmini", "PrecServer", "Etc1", "Etc2")

BOT_BUCKET = "bot_data"
PDF_DIR    = "pdf"
BACKUP_DIR = "backup"

def _secrets(section: str) -> dict:
    try:
        return dict(st.secrets.get(section, {}))
    except Exception:
        return {}

def _pick(*candidates, default=None):
    for v in candidates:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return default

def _resolve_root(spec: str | None, *, app_root: Path, mounts: dict, default_root: Path) -> Path:
    """
    "project:<rel>" -> app_root/<rel>
    "mount:<Name>/<sub>" -> mounts[Name]/<sub>
    "/abs/path" or "relative" -> 絶対に解決（relative は app_root/relative 扱い）
    None/空 -> default_root
    """
    if not spec:
        return default_root

    s = str(spec).strip()
    if s.startswith("project:"):
        rel = s.split(":", 1)[1].strip()
        return (app_root / rel).resolve()

    if s.startswith("mount:"):
        rest = s.split(":", 1)[1].strip()
        if "/" not in rest:
            return default_root
        mname, sub = rest.split("/", 1)
        base = mounts.get(mname)
        if not base:
            return default_root
        return (Path(str(base)).expanduser() / sub).resolve()

    p = Path(s)
    if not p.is_absolute():
        p = (app_root / p)
    return p.resolve()

@dataclass
class PathConfig:
    app_root: Path
    preset: str
    ssd_path: Path
    pdf_root: Path
    backup_root: Path
    vs_root: Path

    @classmethod
    def load(cls, app_root: Path) -> "PathConfig":
        env_sec    = _secrets("env")
        mounts_sec = _secrets("mounts")
        paths_sec  = _secrets("paths")   # 互換: 直接 vs_root/pdf_root/backup_root を置ける
        pdf_sec    = _secrets("pdf")     # 最優先
        backup_sec = _secrets("backup")  # 最優先

        # 1) location 決定
        preset = _pick(
            env_sec.get("location"),
            os.getenv("APP_LOCATION_PRESET"),
            "Home",
        )
        if preset not in AVAILABLE_PRESETS:
            raise ValueError(f"Unknown location preset: {preset}. Allowed: {AVAILABLE_PRESETS}")

        # 2) ssd_path（mount 必須）
        if preset not in mounts_sec:
            raise ValueError(f"location={preset} に対応する mount が secrets.toml の [mounts] にありません。")
        ssd_path = Path(str(mounts_sec[preset])).expanduser()

        # 3) 既定ルート（mount/bot_data/...）
        default_pdf_root = Path(_pick(
            paths_sec.get("pdf_root"),
            os.getenv("APP_PDF_ROOT"),
            ssd_path / BOT_BUCKET / PDF_DIR,
        )).expanduser()

        default_backup_root = Path(_pick(
            paths_sec.get("backup_root"),
            os.getenv("APP_BACKUP_ROOT"),
            ssd_path / BOT_BUCKET / BACKUP_DIR,
        )).expanduser()

        # 4) VS 出力（プロジェクト内既定）
        vs_root = Path(_pick(
            paths_sec.get("vs_root"),
            os.getenv("APP_VS_ROOT"),
            app_root / "data" / "vectorstore",
        )).expanduser()

        # 5) 最優先セクションで最終決定
        pdf_root = _resolve_root(pdf_sec.get("root"),   app_root=app_root, mounts=mounts_sec, default_root=default_pdf_root)
        backup_root = _resolve_root(backup_sec.get("root"), app_root=app_root, mounts=mounts_sec, default_root=default_backup_root)

        cfg = cls(
            app_root=app_root,
            preset=preset,
            ssd_path=ssd_path,
            pdf_root=pdf_root,
            backup_root=backup_root,
            vs_root=vs_root,
        )
        cfg.ensure_dirs()
        return cfg

    def ensure_dirs(self):
        # 出力系は事前に作成
        self.vs_root.mkdir(parents=True, exist_ok=True)
        self.backup_root.mkdir(parents=True, exist_ok=True)

def resolve_paths_for(preset: str, app_root: Path) -> PathConfig:
    """UI から一時切替。PDF/Backup も [pdf]/[backup] の指定を最優先で解釈。"""
    if preset not in AVAILABLE_PRESETS:
        raise ValueError(f"Unknown location preset: {preset}. Allowed: {AVAILABLE_PRESETS}")

    mounts_sec = _secrets("mounts")
    pdf_sec    = _secrets("pdf")
    backup_sec = _secrets("backup")

    if preset not in mounts_sec:
        raise ValueError(f"[mounts].{preset} が secrets.toml にありません。")

    ssd_path = Path(str(mounts_sec[preset])).expanduser()

    default_pdf_root   = (ssd_path / BOT_BUCKET / PDF_DIR).expanduser()
    default_backup_root= (ssd_path / BOT_BUCKET / BACKUP_DIR).expanduser()
    vs_root            = (app_root / "data" / "vectorstore").expanduser()

    pdf_root = _resolve_root(pdf_sec.get("root"),   app_root=app_root, mounts=mounts_sec, default_root=default_pdf_root)
    backup_root = _resolve_root(backup_sec.get("root"), app_root=app_root, mounts=mounts_sec, default_root=default_backup_root)

    cfg = PathConfig(
        app_root=app_root,
        preset=preset,
        ssd_path=ssd_path,
        pdf_root=pdf_root,
        backup_root=backup_root,
        vs_root=vs_root,
    )
    cfg.ensure_dirs()
    return cfg

# ---- グローバル（secrets の location を既定に採用）----
APP_ROOT = Path(__file__).resolve().parents[1]
PATHS = PathConfig.load(APP_ROOT)
