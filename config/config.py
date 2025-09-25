from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import os
import streamlit as st

# ============================================================
# 設定ポリシー（シンプル版）
# - location 候補は AVAILABLE_PRESETS に固定
# - 実マウントパスは secrets.toml の [mounts] からのみ解決（ハードコードの既定は持たない）
# - 既定の location は secrets.toml の [env].location（なければ "Home"）
# - VS 出力はプロジェクト内 data/vectorstore を既定（secrets/env で上書き可）
# - UI からの一時切替用に resolve_paths_for(preset, app_root) を提供
# ============================================================

AVAILABLE_PRESETS = ("Home", "Portable", "PrecMacmini", "PrecServer", "Etc1", "Etc2")

BOT_BUCKET = "bot_data"
PDF_DIR    = "pdf"
BACKUP_DIR = "backup"

def _secrets(section: str) -> dict:
    return dict(st.secrets.get(section, {})) if hasattr(st, "secrets") else {}

def _pick(*candidates, default=None):
    """最初に見つかった有効値を返す（None/空文字はスキップ）"""
    for v in candidates:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return default

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
        env_sec    = _secrets("env")     # [env] location="Home"
        mounts_sec = _secrets("mounts")  # [mounts] Home="/Volumes/Extreme SSD" など
        paths_sec  = _secrets("paths")   # 任意: [paths] で個別上書き

        # 1) location（プリセット）を決定
        preset = _pick(
            env_sec.get("location"),
            os.getenv("APP_LOCATION_PRESET"),
            "Home",
        )
        if preset not in AVAILABLE_PRESETS:
            raise ValueError(f"Unknown location preset: {preset}. Allowed: {AVAILABLE_PRESETS}")

        # 2) ssd_path は secrets.toml の [mounts] に必須
        if preset not in mounts_sec:
            raise ValueError(f"location={preset} に対応する mount が secrets.toml の [mounts] にありません。")
        ssd_path = Path(str(mounts_sec[preset])).expanduser()

        # 3) 入力/バックアップは ssd_path/bot_data/...
        pdf_root = Path(_pick(
            paths_sec.get("pdf_root"),
            os.getenv("APP_PDF_ROOT"),
            ssd_path / BOT_BUCKET / PDF_DIR,
        )).expanduser()

        backup_root = Path(_pick(
            paths_sec.get("backup_root"),
            os.getenv("APP_BACKUP_ROOT"),
            ssd_path / BOT_BUCKET / BACKUP_DIR,
        )).expanduser()

        # 4) VS 出力（プロジェクト内を既定）
        vs_root = Path(_pick(
            paths_sec.get("vs_root"),
            os.getenv("APP_VS_ROOT"),
            app_root / "data" / "vectorstore",
        )).expanduser()

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
        # 入力側は存在チェックのみ、出力側は作成
        self.vs_root.mkdir(parents=True, exist_ok=True)

def resolve_paths_for(preset: str, app_root: Path) -> PathConfig:
    """
    UI からの一時切替用。secrets.toml の [mounts] から ssd_path を取り、
    bot_data/pdf, bot_data/backup を組み立て。vs_root はプロジェクト内既定を使用。
    """
    if preset not in AVAILABLE_PRESETS:
        raise ValueError(f"Unknown location preset: {preset}. Allowed: {AVAILABLE_PRESETS}")

    mounts_sec = _secrets("mounts")
    if preset not in mounts_sec:
        raise ValueError(f"[mounts].{preset} が secrets.toml にありません。")

    ssd_path = Path(str(mounts_sec[preset])).expanduser()
    pdf_root = (ssd_path / BOT_BUCKET / PDF_DIR).expanduser()
    backup_root = (ssd_path / BOT_BUCKET / BACKUP_DIR).expanduser()
    vs_root = (app_root / "data" / "vectorstore").expanduser()

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

# ---- グローバルロード（secrets.toml の location を既定に採用）----
APP_ROOT = Path(__file__).resolve().parents[1]
PATHS = PathConfig.load(APP_ROOT)
