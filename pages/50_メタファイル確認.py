# pages/50_メタファイル確認.py
# ------------------------------------------------------------
# 🗂️ vectorstore/(openai|local)/<shard_id>/meta.jsonl ビューア（シャード対応）
# - 複数シャードの meta.jsonl を横断して読み込み・絞り込み・CSV出力
# - 単一シャード選択時: バックアップ / 選択ファイル削除 / 完全初期化 / 復元
# - 📋 ファイル名コピー: year/file.pdf を1クリックで clipboard にコピー
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple, Any
import json
import shutil
import re

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# 基本パス
# ============================================================
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
VS_ROOT  = DATA_DIR / "vectorstore"

# ============================================================
# JSONL 読み込み
# ============================================================
def iter_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                st.warning(f"[{path.name}] {i} 行目 JSONL パース失敗: {e}")

# ============================================================
# バックアップ関連
# ============================================================
def ensure_backup_dir(base: Path) -> Path:
    bdir = base / "backup" / datetime.now().strftime("%Y%m%d-%H%M%S")
    bdir.mkdir(parents=True, exist_ok=True)
    return bdir

def safe_copy(src: Path, dst_dir: Path) -> Optional[Path]:
    if src.exists():
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return dst
    return None

def backup_all(base_dir: Path) -> Tuple[List[Path], Path]:
    bdir = ensure_backup_dir(base_dir)
    copied: List[Path] = []
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        p = base_dir / name
        dst = safe_copy(p, bdir)
        if dst is not None:
            copied.append(dst)
    return copied, bdir

def list_backup_dirs(base_dir: Path) -> List[Path]:
    broot = base_dir / "backup"
    if not broot.exists():
        return []
    dirs = [p for p in broot.iterdir() if p.is_dir()]
    def sort_key(p: Path):
        try:
            return datetime.strptime(p.name, "%Y%m%d-%H%M%S")
        except Exception:
            return datetime.fromtimestamp(p.stat().st_mtime)
    return sorted(dirs, key=sort_key, reverse=True)

def preview_backup(backup_dir: Path) -> pd.DataFrame:
    rows = []
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        p = backup_dir / name
        rows.append({
            "name": name,
            "exists": "✅" if p.exists() else "❌",
            "size": file_size_readable(p) if p.exists() else "N/A",
            "path": str(p)
        })
    return pd.DataFrame(rows)

def restore_from_backup(base_dir: Path, backup_dir: Path) -> Tuple[List[str], List[str]]:
    restored, missing = [], []
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        src = backup_dir / name
        dst = base_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            restored.append(str(dst))
        else:
            missing.append(str(src))
    return restored, missing

# ============================================================
# processed_files.json ユーティリティ
# ============================================================
def load_processed_files(p: Path) -> Optional[List[Any]]:
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        st.warning(f"processed_files.json 読込失敗: {e}")
        return None

def save_processed_files(p: Path, data: List[Any]) -> None:
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_name(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        for k in ("file", "filename", "path", "name"):
            v = item.get(k)
            if isinstance(v, str):
                return v
        return None
    if isinstance(item, str):
        return item
    return None

def filter_processed_list(items: List[Any], remove_files: List[str]) -> Tuple[List[Any], int]:
    rset = set(remove_files)
    kept, removed = [], 0
    for it in items:
        cand = extract_name(it)
        if isinstance(cand, str) and cand in rset:
            removed += 1
        else:
            kept.append(it)
    return kept, removed

def file_size_readable(p: Path) -> str:
    try:
        n = p.stat().st_size
    except Exception:
        return "N/A"
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} PB"

# ============================================================
# クリップボードコピー（JS埋め込み）
# ============================================================
def copy_button(text: str, label: str, key: str):
    """
    ブラウザの clipboard API を使って text をコピーするボタンを描画。
    text は JSON エスケープして安全に埋め込む。
    """
    payload = json.dumps(text, ensure_ascii=False)
    html = f"""
    <button id="{key}" style="
        padding:6px 10px;border-radius:8px;border:1px solid #dadce0;
        background:#fff;cursor:pointer;font-size:0.9rem;">📋 {label}</button>
    <script>
      const btn = document.getElementById("{key}");
      if (btn) {{
        btn.addEventListener("click", async () => {{
          try {{
            await navigator.clipboard.writeText({payload});
            const old = btn.innerText;
            btn.innerText = "✅ コピーしました";
            setTimeout(()=>{{ btn.innerText = old; }}, 1200);
          }} catch(e) {{
            console.error(e);
            alert("コピーに失敗しました: " + e);
          }}
        }});
      }}
    </script>
    """
    st.components.v1.html(html, height=38)

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="50 メタファイル確認（シャード対応）", page_icon="🗂️", layout="wide")
st.title("🗂️ メタファイル確認（フォルダー＝シャード）")

# --- サイドバー：読み込み設定 ---
with st.sidebar:
    st.header("読み込み設定")
    backend = st.radio("バックエンド", ["openai", "local"], index=0, horizontal=True)
    base_backend_dir = VS_ROOT / backend
    if not base_backend_dir.exists():
        st.error(f"vectorstore/{backend} が見つかりません。先にベクトル化を実行してください。")
        st.stop()

    shard_dirs = sorted([p for p in base_backend_dir.iterdir() if p.is_dir()])
    shard_ids = [p.name for p in shard_dirs]
    sel_shards = st.multiselect("対象シャード（複数可）", shard_ids, default=shard_ids)
    max_rows_read = st.number_input("各シャードの読み込み最大行数（0=全件）", min_value=0, value=0, step=1000)

# --- データ読み込み ---
if not sel_shards:
    st.warning("少なくとも1つのシャードを選択してください。")
    st.stop()

all_rows: List[Dict] = []
for sid in sel_shards:
    bdir = base_backend_dir / sid
    meta_path = bdir / "meta.jsonl"
    read_cnt = 0
    for obj in iter_jsonl(meta_path):
        obj = dict(obj)
        obj.setdefault("shard_id", sid)
        obj.setdefault("file", obj.get("doc_id"))
        all_rows.append(obj)
        read_cnt += 1
        if max_rows_read and read_cnt >= max_rows_read:
            break

if not all_rows:
    st.warning("表示できるレコードがありません。")
    st.stop()

df = pd.DataFrame(all_rows)
for col in ["file","page","chunk_id","chunk_index","text","span_start","span_end","shard_id","year"]:
    if col not in df.columns:
        df[col] = None
df["chunk_len"] = df["text"].astype(str).str.len()

# 先頭サンプル表示
st.dataframe(df.head(500), use_container_width=True, height=560)
st.divider()

# ============================================================
# 📋 ファイル名コピー（year/file.pdf をワンクリップ）
# ============================================================
st.subheader("📋 参照用ファイル名コピー（year/file.pdf）")
st.caption("09_ボット.py の『参照ファイル』欄や [[files: ...]] に貼るのに便利です。")

# 一意なファイル一覧（選択シャード内）
file_list = sorted(df["file"].dropna().unique().tolist())

# クイック検索
colf1, colf2 = st.columns([2,1])
with colf1:
    q = st.text_input("ファイル名フィルタ（部分一致）", value="")
with colf2:
    show_count = st.number_input("表示件数", min_value=10, max_value=1000, value=100, step=10)

if q:
    q_norm = q.strip()
    filtered = [f for f in file_list if q_norm.lower() in str(f).lower()]
else:
    filtered = file_list

st.caption(f"ヒット: {len(filtered)} 件")
if not filtered:
    st.info("フィルタに一致するファイルがありません。")
else:
    # 行グリッドでコピーしやすく並べる
    cols = st.columns(3)
    for i, f in enumerate(filtered[:int(show_count)]):
        with cols[i % 3]:
            st.write(f"`{f}`")
            copy_button(text=f, label="year/file.pdf をコピー", key=f"copy_{i}")

st.divider()

# ============================================================
# メンテナンス（単一シャードのみ）
# ============================================================
st.subheader("🛠 メンテナンス（単一シャード選択時のみ）")
if len(sel_shards) != 1:
    st.info("バックアップ / 削除 / 復元は **単一シャード**を選択したときのみ有効です。")
    st.stop()

shard_id = sel_shards[0]
base_dir = base_backend_dir / shard_id
meta_path = base_dir / "meta.jsonl"
vec_path  = base_dir / "vectors.npy"
pf_path   = base_dir / "processed_files.json"

m1, m2 = st.columns(2)

# --- 📦 バックアップ ---
with m1:
    st.markdown("### 📦 バックアップを作成")
    if st.button("📦 バックアップを作成", use_container_width=True):
        copied, bdir = backup_all(base_dir)
        st.success(f"バックアップ完了: {bdir}\n" + "\n".join(f"- {p}" for p in copied))

# --- 🧹 選択ファイルを削除 ---
with m2:
    st.markdown("### 🧹 選択ファイルを削除（このシャードのみ）")
    st.caption("絞り込みで選んだファイルを削除。実行前にバックアップ推奨。")

    sel_files = sorted(df["file"].dropna().unique().tolist())
    target_files = st.multiselect("削除対象ファイル", sel_files)

    reset_pf_file = st.checkbox("processed_files.json もリセット（削除）", value=True)
    confirm_del = st.checkbox("バックアップ済みで削除に同意します。")

    if st.button("🧹 削除を実行", type="primary", use_container_width=True,
                 disabled=not (target_files and confirm_del)):
        try:
            # バックアップ
            copied, bdir = backup_all(base_dir)

            # meta.jsonl 再構築 & vectors.npy 更新
            keep_lines, keep_vec_indices = [], []
            removed_meta, valid_idx = 0, 0
            with meta_path.open("r", encoding="utf-8") as f:
                for raw in f:
                    s = raw.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except:
                        keep_lines.append(raw)
                        continue
                    fname = obj.get("file") if isinstance(obj, dict) else None
                    if fname in set(target_files):
                        removed_meta += 1
                        valid_idx += 1
                        continue
                    else:
                        keep_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")
                        keep_vec_indices.append(valid_idx)
                        valid_idx += 1
            with meta_path.open("w", encoding="utf-8") as f:
                f.writelines(keep_lines)

            removed_vecs = 0
            if vec_path.exists():
                vecs = np.load(vec_path)
                if vecs.ndim == 2:
                    new_vecs = vecs[keep_vec_indices] if keep_vec_indices else np.empty((0, vecs.shape[1]))
                    removed_vecs = vecs.shape[0] - new_vecs.shape[0]
                    np.save(vec_path, new_vecs)

            if reset_pf_file:
                if pf_path.exists():
                    pf_path.unlink()
            else:
                pf_list = load_processed_files(pf_path)
                if pf_list is not None:
                    kept, _ = filter_processed_list(pf_list, target_files)
                    save_processed_files(pf_path, kept)

            st.success(
                f"削除完了 ✅\n- meta.jsonl: {removed_meta} 行削除\n- vectors.npy: {removed_vecs} 行削除\n"
                + ("- processed_files.json: 削除（リセット）" if reset_pf_file else "- processed_files.json: 対象だけ除外")
                + f"\n- バックアップ: {bdir}"
            )
        except Exception as e:
            st.error(f"削除中にエラー: {e}")

st.divider()

# --- 🗑️ 完全初期化 ---
st.subheader("🗑️ このシャードを完全初期化")
st.caption("meta.jsonl / vectors.npy / processed_files.json を全部削除します。必ずバックアップしてから。")
confirm_wipe = st.checkbox("完全初期化に同意します。")
if st.button("🗑️ 完全初期化（このシャード）", type="secondary", use_container_width=True, disabled=not confirm_wipe):
    deleted = []
    for name in ["meta.jsonl","vectors.npy","processed_files.json"]:
        p = base_dir / name
        if p.exists():
            p.unlink()
            deleted.append(str(p))
    if deleted:
        st.success("削除しました:\n" + "\n".join(f"- {x}" for x in deleted))
