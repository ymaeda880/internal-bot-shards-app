# lib/vectorstore_utils.py
# ------------------------------------------------------------
# vectorstore 関連の共通ユーティリティ
# - meta.jsonl / vectors.npy / processed_files.json の管理
# - バックアップ（APP_ROOT/backup 配下に統一）/ 復元 / 削除支援
# - バックアップ年齢チェック / 条件付きバックアップ
# - 追加: vectors.npy shape(mmap) / JSONL行数 / RAM見積 / シャード走査
# - Streamlit 非依存でも動作（警告は print にフォールバック）
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple, Any
import json
import shutil
import numpy as np

# streamlit を任意依存に（無い環境でも OK）
try:
    import streamlit as st
    def _warn(msg: str) -> None:
        try:
            st.warning(msg)
        except Exception:
            print(f"[WARN] {msg}")
except Exception:
    st = None  # type: ignore
    def _warn(msg: str) -> None:
        print(f"[WARN] {msg}")

# ------------------------------------------------------------
# ルート系
# ------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parents[1]
BACKUP_ROOT = APP_ROOT / "backup"

__all__ = [
    # JSONL
    "iter_jsonl", "count_jsonl_lines",
    # vectors / meta
    "load_vector_shape", "estimate_ram_gb",
    # バックアップ
    "ensure_backup_dir", "backup_all", "list_backup_dirs", "latest_backup_dir",
    "backup_age_days", "preview_backup", "restore_from_backup",
    # processed_files
    "load_processed_files", "save_processed_files", "extract_name",
    "filter_processed_list",
    # 便利関数
    "file_size_readable", "scan_shards",
]

# ============================================================
# JSONL 読み込み・行数
# ============================================================
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    JSONL を1行ずつ yield（壊れ行は警告してスキップ）
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
                else:
                    _warn(f"[{path.name}] {i} 行目: dict 以外の JSON はスキップ")
            except Exception as e:
                _warn(f"[{path.name}] {i} 行目 JSONL パース失敗: {e}")

def count_jsonl_lines(path: Path) -> int:
    """
    JSONL の総行数を返す（ファイル無ければ 0）
    """
    if not path.exists():
        return 0
    c = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            c += 1
    return c

# ============================================================
# vectors.npy（shape 取得は mmap）
# ============================================================
def load_vector_shape(vec_path: Path) -> Tuple[int, int]:
    """
    vectors.npy の shape=(n, d) を mmap で安全に取得。
    想定外 shape は (len, 1) や (0, 0) に丸めて返す。
    """
    arr = np.load(vec_path, mmap_mode="r")
    shp = tuple(arr.shape)
    if len(shp) == 2:
        return int(shp[0]), int(shp[1])
    if len(shp) == 1:
        return int(shp[0]), 1
    return 0, 0

def estimate_ram_gb(n: int, d: int, dtype_bytes: int = 4) -> float:
    """
    float32=4 bytes を既定とした RAM 見積（GB）
    """
    return (n * d * dtype_bytes) / (1024**3)

# ============================================================
# バックアップ関連（APP_ROOT/backup/<backend>/<shard_id>/<timestamp>/）
# ============================================================
def _parse_ts(name: str) -> Optional[datetime]:
    try:
        # 例: 20250923-125500（UTC）
        return datetime.strptime(name, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def ensure_backup_dir(backend: str, shard_id: str) -> Path:
    """
    共通 backup フォルダ直下に backend/shard_id/timestamp/ を作成して返す
    """
    base = BACKUP_ROOT / backend / shard_id
    base.mkdir(parents=True, exist_ok=True)
    bdir = base / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    bdir.mkdir(parents=True, exist_ok=True)
    return bdir

def file_size_readable(p: Path) -> str:
    """
    ファイルサイズを 1桁小数（KB, MB, GB, TB, PB）で返す
    """
    try:
        n = float(p.stat().st_size)
    except Exception:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    for u in units:
        if n < 1024.0 or u == units[-1]:
            return f"{n:,.1f} {u}"
        n /= 1024.0
    return f"{n:,.1f} PB"

def backup_all(base_dir: Path, backend: str, shard_id: str) -> Tuple[List[Path], Path]:
    """
    base_dir の meta.jsonl / vectors.npy / processed_files.json を
    APP_ROOT/backup/<backend>/<shard_id>/<timestamp>/ にコピー
    """
    bdir = ensure_backup_dir(backend, shard_id)
    copied: List[Path] = []
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        src = base_dir / name
        if src.exists():
            dst = bdir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)
    # 何もコピーされなかった時も “空バックアップ”を残す方針
    return copied, bdir

def list_backup_dirs(backend: str, shard_id: str) -> List[Path]:
    """
    APP_ROOT/backup/<backend>/<shard_id>/ 直下のタイムスタンプフォルダ一覧（新しい順）
    """
    broot = BACKUP_ROOT / backend / shard_id
    if not broot.exists():
        return []
    dirs = [p for p in broot.iterdir() if p.is_dir()]
    def sort_key(p: Path):
        ts = _parse_ts(p.name)
        try:
            return ts or datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return datetime.fromtimestamp(0, tz=timezone.utc)
    return sorted(dirs, key=sort_key, reverse=True)

def latest_backup_dir(backend: str, shard_id: str) -> Optional[Path]:
    lst = list_backup_dirs(backend, shard_id)
    return lst[0] if lst else None

def backup_age_days(backend: str, shard_id: str) -> Optional[float]:
    """
    直近バックアップからの経過日数（小数）を返す。バックアップがなければ None。
    """
    latest = latest_backup_dir(backend, shard_id)
    if not latest:
        return None
    ts = _parse_ts(latest.name)
    if not ts:
        # タイムスタンプ形式でない場合は mtime で代替
        try:
            ts = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return None
    now = datetime.now(timezone.utc)
    return (now - ts).total_seconds() / 86400.0

# DataFrame 表示は UI 側で使いやすいよう情報を整形して返す
def preview_backup(backup_dir: Path):
    rows = []
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        p = backup_dir / name
        rows.append({
            "name": name,
            "exists": "✅" if p.exists() else "❌",
            "size": file_size_readable(p) if p.exists() else "N/A",
            "path": str(p)
        })
    try:
        import pandas as pd  # 遅延 import（pandas 非依存環境を邪魔しない）
        return pd.DataFrame(rows)
    except Exception:
        return rows  # pandas 無ければ list[dict] を返す

def restore_from_backup(base_dir: Path, backup_dir: Path) -> Tuple[List[str], List[str]]:
    """
    バックアップから base_dir へ復元。存在するものだけコピー。
    戻り値: (復元できたパス, 見つからなかったパス)
    """
    restored, missing = [], []
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        src = backup_dir / name
        dst = base_dir / name
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(str(dst))
        else:
            missing.append(str(src))
    return restored, missing

# ============================================================
# processed_files.json ユーティリティ
# ============================================================
def load_processed_files(p: Path) -> Optional[List[Any]]:
    """
    processed_files.json を読み込み list を返す（壊れていたら None）
    """
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        _warn(f"processed_files.json 読込失敗: {e}")
        return None

def save_processed_files(p: Path, data: List[Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_name(item: Any) -> Optional[str]:
    """
    processed_files の各要素から “ファイル名 or パス文字列” を抽出
    """
    if isinstance(item, dict):
        for k in ("file", "filename", "path", "name"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return None
    if isinstance(item, str):
        return item.strip() or None
    return None

def _basename_s(s: str) -> str:
    try:
        return Path(s).name
    except Exception:
        return s

def filter_processed_list(items: List[Any], remove_files: List[str]) -> Tuple[List[Any], int]:
    """
    渡された processed 配列から remove_files に一致するものを取り除く。
    - パス/名前混在を吸収するため、basename で比較。
    戻り値: (残した配列, 削除数)
    """
    rset = {_basename_s(x) for x in remove_files if isinstance(x, str)}
    kept, removed = [], 0
    for it in items:
        cand = extract_name(it)
        if isinstance(cand, str) and _basename_s(cand) in rset:
            removed += 1
        else:
            kept.append(it)
    return kept, removed

# ============================================================
# 便利：シャード走査（健全性ダッシュボード等で再利用）
# ============================================================
def scan_shards(backend_root: Path) -> List[Dict[str, Any]]:
    """
    data/vectorstore/<backend>/ 配下の shard を走査し、主要統計を返す
    """
    if not backend_root.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for shp in sorted([p for p in backend_root.iterdir() if p.is_dir()]):
        vec = shp / "vectors.npy"
        meta = shp / "meta.jsonl"
        if not vec.exists():
            continue
        try:
            n, d = load_vector_shape(vec)
        except Exception as e:
            _warn(f"{shp.name}: vectors.npy 読み込みエラー: {e}")
            n, d = 0, 0
        size_gb = (vec.stat().st_size / (1024**3)) if vec.exists() else 0.0
        meta_rows = count_jsonl_lines(meta)
        rows.append({
            "shard_id": shp.name,
            "n_vectors": n,
            "dim": d,
            "vectors_gb": size_gb,
            "meta_rows": meta_rows,
            "mismatch": n - meta_rows,
            "est_ram_gb": estimate_ram_gb(n, d),
            "path": str(shp),
        })
    return rows
