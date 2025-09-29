# config/path_config.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence
import os

# ============================================================
# パス設定（settings.toml 版・簡素化）
# - 読み取り元: config/settings.toml
#   [env]        : location = "Develop" | "Home" | "Server"
#   [mounts]     : Home="/Volumes/Extreme SSD", Server="/srv/ssd" など
#   [locations.*]: 環境ごとのルート（pdf_root / backup_root）
#   [app]        : available_presets=["Develop","Home","Server"], bot_bucket="bot_data"
# - フォールバック:
#   available_presets = ("Develop","Home","Server")
#   bot_bucket = "bot_data"
#   vs_root    = <APP_ROOT>/data/vectorstore（環境変数 APP_VS_ROOT で上書き可）
#   pdf_root   = <APP_ROOT>/pdf（locations 未定義時）
#   backup_root= <APP_ROOT>/backup（locations 未定義時）
# - 設定ファイルは APP_SETTINGS_FILE でも指定可能
# ============================================================

# --- toml loader (3.11+: tomllib / fallback: tomli) ---
try:
    import tomllib as _toml  # Python 3.11+
except Exception:
    try:
        import tomli as _toml  # type: ignore
    except Exception:
        _toml = None  # toml 読み込み不可（空設定として扱う）

APP_ROOT = Path(__file__).resolve().parents[1]


def _pick(*candidates, default=None):
    """最初に見つかった有効値（None/空文字以外）を返す。"""
    for v in candidates:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return default


def _load_toml(path: Path) -> Dict[str, Any]:
    """TOML を辞書で返す。存在しない/読めない場合は空 dict。"""
    if _toml is None or not path.exists() or not path.is_file():
        return {}
    try:
        with path.open("rb") as f:
            return dict(_toml.load(f))
    except Exception:
        return {}


def _settings_file() -> Path:
    """設定ファイルの探索順: APP_SETTINGS_FILE → APP_ROOT/config/settings.toml"""
    env = os.getenv("APP_SETTINGS_FILE")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = (APP_ROOT / p)
        return p.resolve()
    return (APP_ROOT / "config" / "settings.toml").resolve()


def _parse_list_from_env(var: str) -> Optional[Sequence[str]]:
    """カンマ区切りの環境変数を配列に。未設定なら None。"""
    s = os.getenv(var)
    if not s:
        return None
    items = [x.strip() for x in s.split(",")]
    return [x for x in items if x]


def _resolve_root(spec: str | None, *, mounts: dict, default_root: Path) -> Path:
    """spec を実パスに解決。相対は APP_ROOT 基準。"""
    if not spec:
        return default_root

    s = str(spec).strip()
    if s.startswith("project:"):
        rel = s.split(":", 1)[1].strip()
        return (APP_ROOT / rel).resolve()

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
        p = (APP_ROOT / p)
    return p.resolve()


@dataclass
class PathConfig:
    preset: str
    app_root: Path
    ssd_path: Path          # mounts[preset] が無ければ APP_ROOT
    pdf_root: Path
    backup_root: Path
    vs_root: Path

    @classmethod
    def load(cls) -> "PathConfig":
        # 設定読み込み
        settings_path = _settings_file()
        settings = _load_toml(settings_path)

        env_sec    = dict(settings.get("env", {}))
        mounts_sec = dict(settings.get("mounts", {}))
        locs_sec   = dict(settings.get("locations", {}))
        app_sec    = dict(settings.get("app", {}))

        # ---- [app] の読み込み + フォールバック/ENV 上書き ----
        env_presets = _parse_list_from_env("APP_AVAILABLE_PRESETS")
        available_presets = tuple(
            env_presets
            or app_sec.get("available_presets")
            or ("Develop", "Home", "Server")
        )

        bot_bucket = str(_pick(
            os.getenv("APP_BOT_BUCKET"),
            app_sec.get("bot_bucket"),
            "bot_data",
        ))

        # 1) location
        preset = _pick(
            env_sec.get("location"),
            os.getenv("APP_LOCATION_PRESET"),
            "Develop",
        )
        if preset not in available_presets:
            raise ValueError(
                f"Unknown location preset: {preset}. "
                f"Allowed: {available_presets}"
            )

        # 2) ssd_path（存在しない場合は APP_ROOT にフォールバック）
        ssd_path = Path(str(mounts_sec.get(preset, APP_ROOT))).expanduser()

        # 3) VS 出力（プロジェクト内既定 or 環境変数）
        vs_root = Path(_pick(
            os.getenv("APP_VS_ROOT"),
            APP_ROOT / "data" / "vectorstore",
        )).expanduser()

        # 4) 現在プリセットの locations を取得
        cur_loc = dict(locs_sec.get(preset, {}))

        # 5) 既定ルート（locations 未定義時のフォールバック）
        default_pdf_root    = (APP_ROOT / "pdf").resolve()
        default_backup_root = (APP_ROOT / "backup").resolve()

        # 6) 解決（project:/mount:/相対/絶対）
        pdf_root    = _resolve_root(cur_loc.get("pdf_root"),    mounts=mounts_sec, default_root=default_pdf_root)
        backup_root = _resolve_root(cur_loc.get("backup_root"), mounts=mounts_sec, default_root=default_backup_root)

        cfg = cls(
            preset=preset,
            app_root=APP_ROOT,
            ssd_path=ssd_path,
            pdf_root=pdf_root,
            backup_root=backup_root,
            vs_root=vs_root,
        )
        cfg.ensure_dirs()
        return cfg

    def ensure_dirs(self):
        # 書き込み不能な場所でも落ちないように保護
        for p in (self.vs_root, self.backup_root):
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    def __str__(self) -> str:
        lines = [
            "=== PathConfig ===",
            f" preset      : {self.preset}",
            f" app_root    : {self.app_root}",
            f" ssd_path    : {self.ssd_path}",
            f" pdf_root    : {self.pdf_root}",
            f" backup_root : {self.backup_root}",
            f" vs_root     : {self.vs_root}",
        ]
        return "\n".join(lines)


# ---- グローバルインスタンス ----
PATHS = PathConfig.load()

if __name__ == "__main__":
    try:
        print(PATHS)
    except Exception as e:
        print("[path_config] エラー:", e)


# python config/path_config.py

# === PathConfig ===
#  preset      : Develop
#  app_root    : /Users/macmini2025/myVenv/myProject/internal_bot_shards_app
#  ssd_path    : /Users/macmini2025/myVenv/myProject/internal_bot_shards_app
#  pdf_root    : /Users/macmini2025/myVenv/myProject/internal_bot_shards_app/pdf
#  backup_root : /Users/macmini2025/myVenv/myProject/internal_bot_shards_app/backup
#  vs_root     : /Users/macmini2025/myVenv/myProject/internal_bot_shards_app/data/vectorstore

