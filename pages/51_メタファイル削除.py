# pages/51_メタファイル削除.py
# ------------------------------------------------------------
# 🗑️ メタファイル削除ページ
# - 削除 / 初期化 / バックアップ / 復元
# - バックアップ保存先：PATHS.backup_root / <backend> / <shard_id> / <timestamp>
# - 追加機能:
#   1) すべてのシャードを即時バックアップ
#   2) 対象シャードのみ即時バックアップ
#   3) 「未バックアップ日数」しきい値で一括バックアップ
#   4) シャードごと削除（フォルダ完全削除→空フォルダ再作成）
#   5) 完全初期化にもバックアップ＆DELETE確認
# ------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import os
import urllib.parse
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

from config.config import PATHS  # ✅ vs_root / backup_root を集中管理
from lib.vectorstore_utils import iter_jsonl  # 既存ユーティリティ

# ============================================================
# 基本パス（config に合わせる）
# ============================================================
VS_ROOT: Path      = PATHS.vs_root
BACKUP_ROOT: Path  = PATHS.backup_root

# ============================================================
# UI 設定
# ============================================================
st.set_page_config(page_title="51 メタファイル削除", page_icon="🗑️", layout="wide")
st.title("🗑️ メタファイル削除（シャード単位）")
st.caption(f"Backup base: `{BACKUP_ROOT}` / VectorStore: `{VS_ROOT}`")
st.info("このページは **削除・初期化・バックアップ/復元** に特化しています。作業前に必ずバックアップを作成してください。")

# ============================================================
# サイドバー: シャード選択
# ============================================================
with st.sidebar:
    st.header("対象シャード")
    backend = st.radio("バックエンド", ["openai", "local"], index=0, horizontal=True, key="sb_backend")
    base_backend_dir = VS_ROOT / backend
    if not base_backend_dir.exists():
        st.error(f"{base_backend_dir} が見つかりません。先にベクトル化を実行してください。")
        st.stop()

    shard_dirs = sorted([p for p in base_backend_dir.iterdir() if p.is_dir()])
    shard_ids = [p.name for p in shard_dirs]
    shard_id = st.selectbox("対象シャード", shard_ids if shard_ids else ["(なし)"], key="sb_shard")
    if not shard_ids:
        st.error("シャードが存在しません。")
        st.stop()

# ============================================================
# 対象シャードのパス
# ============================================================
base_dir = base_backend_dir / shard_id
meta_path = base_dir / "meta.jsonl"
vec_path  = base_dir / "vectors.npy"
pf_path   = base_dir / "processed_files.json"

# ============================================================
# バックアップ（このページで実装）
# ============================================================
def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")

def _backup_dir_for(backend: str, shard_id: str, ts: str | None = None) -> Path:
    if ts is None:
        ts = _timestamp()
    return BACKUP_ROOT / backend / shard_id / ts

def backup_all_local(src_dir: Path, backend: str, shard_id: str) -> tuple[list[str], Path]:
    """
    src_dir（= VS_ROOT/backend/shard）から meta.jsonl / vectors.npy / processed_files.json を
    BACKUP_ROOT/backend/shard/<timestamp>/ にコピー。存在するものだけコピー。
    """
    ts_dir = _backup_dir_for(backend, shard_id)
    ts_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, ts_dir / name)
            copied.append(name)
    return copied, ts_dir

def list_backup_dirs_local(backend: str, shard_id: str) -> list[Path]:
    root = BACKUP_ROOT / backend / shard_id
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)

def preview_backup_local(bdir: Path) -> pd.DataFrame:
    rows = []
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        p = bdir / name
        if p.exists():
            size = p.stat().st_size
            rows.append({"name": name, "size(bytes)": size, "path": str(p)})
    return pd.DataFrame(rows)

def restore_from_backup_local(dst_dir: Path, bdir: Path) -> tuple[list[str], list[str]]:
    restored, missing = [], []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        src = bdir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            restored.append(name)
        else:
            missing.append(name)
    return restored, missing

def backup_age_days_local(backend: str, shard_id: str) -> float | None:
    import time
    bdirs = list_backup_dirs_local(backend, shard_id)
    if not bdirs:
        return None
    latest = bdirs[0]
    mtimes = []
    for name in ["meta.jsonl", "vectors.npy", "processed_files.json"]:
        p = latest / name
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    if not mtimes:
        mtimes.append(latest.stat().st_mtime)
    age_sec = max(time.time() - max(mtimes), 0.0)
    return age_sec / 86400.0

# ============================================================
# processed_files.json 最適化：構造保持での選択削除
#  - 対応スキーマ：
#      * {"done":[...]}
#      * [...]
#      * [{"done":[...]}, {"done":[...]}]   ← よくあるケース
#  - 要素は str / dict（file/path/name/relpath/source/original/orig/pdf）いずれでもOK
#  - 照合は フル/相対(shard/filename)/basename/stem + 正規化（NFKC・URLdecode・区切り統一・lower）
# ============================================================
def _canon(s: str) -> str:
    if not s:
        return ""
    s = urllib.parse.unquote(s)
    s = unicodedata.normalize("NFKC", s).strip()
    s = s.replace("\\", "/")
    s = os.path.normpath(s).replace("\\", "/")
    return s.lower()

def _entry_to_pathlike(entry) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for k in ("file", "path", "name", "relpath", "source", "original", "orig", "pdf"):
            v = entry.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""

def _load_pf_struct(pf_path: Path):
    if not pf_path.exists():
        return "empty", None, []
    try:
        root = json.loads(pf_path.read_text(encoding="utf-8"))
    except Exception:
        return "unknown", None, []

    if isinstance(root, dict) and isinstance(root.get("done"), list):
        return "object_done", root, [root["done"]]

    if isinstance(root, list):
        done_lists = []
        all_str = True
        for e in root:
            if isinstance(e, dict) and isinstance(e.get("done"), list):
                done_lists.append(e["done"])
                all_str = False
            elif not isinstance(e, str):
                all_str = False
        if done_lists:
            return "array_of_done_objects", root, done_lists
        if all_str:
            return "array", root, [root]

    return "unknown", root, []

def _save_pf_struct(pf_path: Path, schema: str, root_obj):
    if schema in ("object_done", "array", "array_of_done_objects") and root_obj is not None:
        pf_path.write_text(json.dumps(root_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    items: list[str] = []
    if isinstance(root_obj, dict) and isinstance(root_obj.get("done"), list):
        for e in root_obj["done"]:
            s = _entry_to_pathlike(e)
            if s:
                items.append(s)
    elif isinstance(root_obj, list):
        for e in root_obj:
            if isinstance(e, dict) and isinstance(e.get("done"), list):
                for x in e["done"]:
                    s = _entry_to_pathlike(x)
                    if s:
                        items.append(s)
            else:
                s = _entry_to_pathlike(e)
                if s:
                    items.append(s)
    items = sorted(set(items))
    pf_path.write_text(json.dumps({"done": items}, ensure_ascii=False, indent=2), encoding="utf-8")

def remove_from_processed_files_selective(pf_path: Path, removed_files: list[str]) -> tuple[int, int, int, list[str]]:
    schema, root, list_refs = _load_pf_struct(pf_path)
    if not list_refs:
        return (0, 0, 0, [])

    t_full = {_canon(x) for x in removed_files}
    t_base = {os.path.basename(x) for x in t_full}
    t_stem = {os.path.splitext(b)[0] for b in t_base}

    def _match(entry) -> bool:
        raw = _entry_to_pathlike(entry)
        cn = _canon(raw)
        if not cn:
            return False
        base = os.path.basename(cn)
        stem = os.path.splitext(base)[0]
        return (
            (cn in t_full) or
            (base in t_base) or
            (stem in t_stem) or
            any(cn.endswith("/" + t) for t in t_full)
        )

    before_total = sum(len(lst) for lst in list_refs)
    removed_show: list[str] = []

    for lst in list_refs:
        new_lst = []
        for e in lst:
            if _match(e):
                raw = _entry_to_pathlike(e)
                removed_show.append(raw if raw else json.dumps(e, ensure_ascii=False)[:120])
            else:
                new_lst.append(e)
        lst.clear()
        lst.extend(new_lst)

    _save_pf_struct(pf_path, schema, root)

    after_total = sum(len(lst) for lst in list_refs)
    removed_count = before_total - after_total
    return (before_total, after_total, removed_count, removed_show[:10])

# ============================================================
# 🔁 バックアップ（拡張機能）
# ============================================================
st.subheader("🛡️ バックアップ（拡張）")

col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("⚡ 対象シャードを即時バックアップ", use_container_width=True, key="bak_one"):
        copied, bdir = backup_all_local(base_dir, backend, shard_id)
        if copied:
            st.success(f"[{backend}/{shard_id}] をバックアップ: {bdir}")
        else:
            st.warning(f"[{backend}/{shard_id}] コピー対象がありません。保存先: {bdir}")

with col_b:
    if st.button("⚡ すべてのシャードを即時バックアップ", use_container_width=True, key="bak_all"):
        summary = []
        for sid in shard_ids:
            sdir = base_backend_dir / sid
            copied, bdir = backup_all_local(sdir, backend, sid)
            summary.append((sid, len(copied), bdir))
        ok = [f"- {sid}: {n}項目 -> {bdir}" for sid, n, bdir in summary]
        st.success("即時バックアップ完了:\n" + "\n".join(ok))

with col_c:
    threshold = st.selectbox("未バックアップ日数 以上ならバックアップ", [1,2,3,7,14,30], index=2, key="bak_thr")
    if st.button("🗓 条件バックアップを実行", use_container_width=True, key="bak_cond"):
        triggered, skipped = [], []
        for sid in shard_ids:
            age = backup_age_days_local(backend, sid)
            if age is None or age >= float(threshold):
                sdir = base_backend_dir / sid
                copied, bdir = backup_all_local(sdir, backend, sid)
                triggered.append((sid, age, len(copied), bdir))
            else:
                skipped.append((sid, age))
        msg = ""
        if triggered:
            msg += "バックアップ実行（閾値超過 or 未実施）:\n" + "\n".join(
                f"- {sid}: age={('None' if age is None else f'{age:.2f}d')} -> {n}項目 @ {bdir}"
                for sid, age, n, bdir in triggered
            )
        if skipped:
            if msg: msg += "\n\n"
            msg += "スキップ（閾値未満）:\n" + "\n".join(f"- {sid}: age={age:.2f}d" for sid, age in skipped)
        st.info(msg or "対象がありませんでした。")

st.divider()

# ============================================================
# プレビュー
# ============================================================
st.subheader("📄 現状プレビュー")
rows = [dict(obj) for obj in iter_jsonl(meta_path)] if meta_path.exists() else []
if not rows:
    st.warning("このシャードには meta.jsonl が存在しないか、レコードがありません。")
else:
    df = pd.DataFrame(rows)
    if "file" not in df.columns:
        df["file"] = None
    st.caption(f"レコード数: {len(df):,}")
    st.dataframe(df.head(500), use_container_width=True, height=420)

st.divider()

# ============================================================
# バックアップ（個別プレビュー）
# ============================================================
st.subheader("📦 バックアップ（個別プレビュー）")
bdirs = list_backup_dirs_local(backend, shard_id)
if bdirs:
    sel_bdir_prev = st.selectbox("バックアッププレビュー", bdirs, format_func=lambda p: p.name, key="prev_bdir")
    if sel_bdir_prev:
        st.dataframe(preview_backup_local(sel_bdir_prev), use_container_width=True, height=180)
else:
    st.caption("まだバックアップがありません。")

st.divider()

# ============================================================
# 選択ファイル削除（processed_files の扱いをラジオで選択）
# ============================================================
st.subheader("🧹 選択ファイル削除")
if rows:
    files = sorted(pd.Series([r.get("file") for r in rows if r.get("file")]).unique().tolist())
    c1, c2 = st.columns([2,1])
    with c1:
        target_files = st.multiselect("削除対象ファイル（year/file.pdf など）", files, key="sel_targets")
    with c2:
        pf_mode = st.radio(
            "processed_files.json の処理",
            ["選択ファイルを消す（既定）", "完全リセット（全削除）", "変更しない"],
            index=0,
            key="pf_mode"
        )
        confirm_del = st.checkbox("削除に同意します", key="confirm_selective")

    if st.button("🧹 削除実行", type="primary", use_container_width=True,
                 disabled=not (target_files and confirm_del), key="btn_selective_delete"):
        try:
            # 直前バックアップ
            copied, bdir = backup_all_local(base_dir, backend, shard_id)

            # meta.jsonl 再構築 + vectors.npy 同期
            keep_lines, keep_vec_indices = [], []
            removed_meta, valid_idx = 0, 0
            target_set = set(target_files)

            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as f:
                    for raw in f:
                        s = raw.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except Exception:
                            keep_lines.append(raw)  # 壊れ行は保全
                            continue
                        fname = obj.get("file") if isinstance(obj, dict) else None
                        if fname in target_set:
                            removed_meta += 1
                            valid_idx += 1
                            continue
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

            # processed_files.json の処理（ラジオ選択）
            pf_msg = ""
            if pf_mode == "完全リセット（全削除）":
                if pf_path.exists():
                    pf_path.unlink()
                pf_msg = "- processed_files.json: 削除（完全リセット）\n"

            elif pf_mode == "選択ファイルを消す（既定）":
                if pf_path.exists():
                    before, after, removed_pf, removed_list = remove_from_processed_files_selective(pf_path, target_files)
                    if removed_pf > 0:
                        st.success(
                            "processed_files.json を更新しました:\n"
                            f"- 除外数: {removed_pf} 件 (before={before}, after={after})\n"
                            f"- 除外された項目の例: {removed_list}"
                        )
                    else:
                        st.warning(
                            "processed_files.json に一致する項目が見つかりませんでした。\n"
                            "（フル/相対/basename/stem・NFKC・URLデコード・区切り統一で照合しています）"
                        )
                else:
                    pf_msg = "- processed_files.json: 見つかりませんでした（処理スキップ）\n"

            else:  # "変更しない"
                pf_msg = "- processed_files.json: 変更なし\n"

            st.success(
                "削除完了 ✅\n"
                f"- meta.jsonl: {removed_meta} 行削除\n"
                f"- vectors.npy: {removed_vecs} 行削除\n"
                f"{pf_msg}"
                f"- バックアップ: {bdir}"
            )
        except Exception as e:
            st.error(f"削除中にエラー: {e}")

st.divider()

# ============================================================
# 🗂️ シャードごと削除（フォルダ単位の完全削除）
# ============================================================
st.subheader("🗂️ シャードごと削除（フォルダ完全削除）")

def _dir_stats(d: Path) -> tuple[int, int]:
    if not d.exists():
        return (0, 0)
    n, total = 0, 0
    for p in d.rglob("*"):
        if p.is_file():
            n += 1
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return n, total

if base_dir.exists():
    cnt, total = _dir_stats(base_dir)
    st.caption(f"このシャードフォルダ配下のファイル数: **{cnt:,}** / 合計サイズ: **{total:,} bytes**")
else:
    st.caption("このシャードフォルダは存在しません。")

colx, coly = st.columns([2,1])
with colx:
    do_backup_before_shard_delete = st.checkbox(
        "削除前に標準バックアップ（meta/vectors/processed）を作成する",
        value=True,
        key="sharddel_backup"
    )
    confirm_shard_del = st.checkbox(
        "シャードごと削除に同意します（元に戻せません）",
        key="sharddel_confirm"
    )
with coly:
    typed = st.text_input("タイプ確認：DELETE と入力", value="", key="sharddel_typed")

if st.button("🗂️ シャードごと削除を実行", type="secondary", use_container_width=True,
             disabled=not (confirm_shard_del and typed.strip().upper() == "DELETE"), key="sharddel_exec"):
    try:
        if do_backup_before_shard_delete and base_dir.exists():
            copied, bdir = backup_all_local(base_dir, backend, shard_id)
            st.info(f"事前バックアップ: {bdir} / コピー: {', '.join(copied) if copied else 'なし'}")

        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        st.success(f"シャード `{backend}/{shard_id}` を削除（フォルダ再作成済み）")
    except Exception as e:
        st.error(f"シャード削除中にエラー: {e}")

st.divider()

# ============================================================
# 🗑️ 完全初期化（3ファイルのみ削除：meta.jsonl / vectors.npy / processed_files.json）
#    - 実行前プレビュー（存在する対象と合計サイズ）
#    - 削除前バックアップ（既定オン）
#    - 二重確認（チェックボックス + DELETE 入力）
# ============================================================
st.subheader("🗑️ 完全初期化")

targets = [
    ("meta.jsonl", meta_path),
    ("vectors.npy", vec_path),
    ("processed_files.json", pf_path),
]
present = [(name, p, (p.stat().st_size if p.exists() and p.is_file() else 0)) for name, p in targets if p.exists()]
total_bytes = sum(s for _, _, s in present)

if present:
    lines = [f"- {name}: {p} ({size:,} bytes)" for name, p, size in present]
    st.caption("削除対象（存在しているもののみ）:\n" + "\n".join(lines))
    st.caption(f"合計サイズ: **{total_bytes:,} bytes**")
else:
    st.caption("削除対象のファイルは見つかりませんでした（meta / vectors / processed）。")

col_init_l, col_init_r = st.columns([2, 1])
with col_init_l:
    do_backup_before_wipe = st.checkbox(
        "削除前に標準バックアップ（meta/vectors/processed）を作成する",
        value=True,
        key="wipe_backup"
    )
    confirm_wipe = st.checkbox(
        "完全初期化に同意します（元に戻せません）",
        key="wipe_confirm"
    )
with col_init_r:
    typed_init = st.text_input("タイプ確認：DELETE と入力", value="", key="wipe_typed")

if st.button(
    "🗑️ 初期化実行",
    type="secondary",
    use_container_width=True,
    disabled=not (confirm_wipe and typed_init.strip().upper() == "DELETE"),
    key="wipe_execute"
):
    try:
        if do_backup_before_wipe:
            copied, bdir = backup_all_local(base_dir, backend, shard_id)
            st.info(f"事前バックアップ: {bdir} / コピー: {', '.join(copied) if copied else 'なし'}")

        deleted = []
        for name, p in targets:
            if p.exists():
                p.unlink()
                deleted.append(f"{name}: {p}")

        if deleted:
            st.success("完全初期化しました:\n" + "\n".join(f"- {x}" for x in deleted))
        else:
            st.info("削除対象のファイルがありませんでした。")
    except Exception as e:
        st.error(f"完全初期化中にエラー: {e}")

st.divider()

# ============================================================
# バックアップ復元（新ルート）
# ============================================================
st.subheader("♻️ バックアップ復元")
bdirs = list_backup_dirs_local(backend, shard_id)
if not bdirs:
    st.info("バックアップがありません。先に『バックアップ作成』を実行してください。")
else:
    sel_bdir_restore = st.selectbox("復元するバックアップを選択", bdirs, format_func=lambda p: p.name, key="restore_bdir")
    if sel_bdir_restore:
        st.dataframe(preview_backup_local(sel_bdir_restore), use_container_width=True, height=160)
        ok_restore = st.checkbox("復元に同意します（現在のファイルは上書きされます）", key="restore_ok")
        if st.button("♻️ 復元実行", type="primary", use_container_width=True, disabled=not ok_restore, key="restore_exec"):
            try:
                restored, missing = restore_from_backup_local(base_dir, sel_bdir_restore)
                msg = "復元完了 ✅\n" + "\n".join(f"- {x}" for x in restored)
                if missing:
                    msg += "\n\n存在しなかったバックアップ項目:\n" + "\n".join(f"- {x}" for x in missing)
                st.success(msg)
            except Exception as e:
                st.error(f"復元中にエラー: {e}")
