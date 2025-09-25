# pages/03_pdfベクトル化.py
# ------------------------------------------------------------
# 📥 <SSD>/bot_data/pdf/<shard> を取り込み、
#    ./data/vectorstore/<backend>/<shard>/ に vectors.npy / meta.jsonl を追記。
#    meta には year / page / embed_model を付与。rag_utils.py の API に準拠。
#    重要: meta.jsonl への追記は NumpyVectorDB.add() が行うため、二重追記はしない。
#    ※ OpenAI の埋め込みモデルは text-embedding-3-large に固定（3072 次元）
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import unicodedata
import re
import json
import os

import streamlit as st
import pdfplumber
import numpy as np
import tiktoken

from config.config import PATHS, AVAILABLE_PRESETS, resolve_paths_for
from config import pricing
from lib.rag_utils import split_text, EmbeddingStore, NumpyVectorDB, ProcessedFilesSimple
from lib.vectorstore_utils import load_processed_files, save_processed_files  # 既存ユーティリティを活用

# ============================================================
# 定数
# ============================================================
OPENAI_EMBED_MODEL = "text-embedding-3-large"  # ← 大固定（3072 次元）

# ============================================================
# 日本語正規化（japanese normalization）
# ============================================================
CJK = r"\u3040-\u309F\u30A0-\u30FF\uFF65-\uFF9F\u4E00-\u9FFF\u3400-\u4DBF"
PUNC = r"、。・，．！？：；（）［］｛｝「」『』〈〉《》【】"
_cjk_cjk_space = re.compile(fr"(?<=[{CJK}])\s+(?=[{CJK}])")
_space_before_punc = re.compile(fr"\s+(?=[{PUNC}])")
_space_after_open = re.compile(fr"(?<=[（［｛「『〈《【])\s+")
_space_before_close = re.compile(fr"\s+(?=[）］｝」』〉》】])")
_multi_space = re.compile(r"[ \t\u3000]+")

def normalize_ja_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = _cjk_cjk_space.sub("", s)
    s = _space_before_punc.sub("", s)
    s = _space_after_open.sub("", s)
    s = _space_before_close.sub("", s)
    s = _multi_space.sub(" ", s)
    return s.strip()

# ============================================================
# tokenizer（large に合わせる）
# ============================================================
enc = tiktoken.encoding_for_model(OPENAI_EMBED_MODEL)
def count_tokens(text: str) -> int:
    return len(enc.encode(text))

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="03 ベクトル化（ページ単位・シャード）", page_icon="🧱", layout="wide")
st.title("🧱 フォルダー（=シャード）ごとのベクトル化（page + year付き）")

# --- サイドバー：location 強調 + ラジオ切替 + 整形表示 ---
with st.sidebar:
    st.subheader("📓 現在のロケーション")
    idx0 = AVAILABLE_PRESETS.index(PATHS.preset) if PATHS.preset in AVAILABLE_PRESETS else 0
    ui_preset = st.radio(
        "Location（この実行中のみ切替）",
        AVAILABLE_PRESETS,
        index=idx0,
        horizontal=True,
        help="secrets.toml の [mounts] で定義されたマウントを使用します。",
    )

EFFECTIVE = resolve_paths_for(ui_preset, PATHS.app_root) if ui_preset != PATHS.preset else PATHS

with st.sidebar:
    st.markdown(f"### 🧭 Location: **{ui_preset}**")
    st.markdown("#### 📂 解決パス（コピー可）")
    st.text_input("ssd_path", str(EFFECTIVE.ssd_path), key="p_ssd", disabled=True)
    st.text_input("PDF_ROOT", str(EFFECTIVE.pdf_root), key="p_pdf", disabled=True)
    st.text_input("BACKUP_ROOT", str(EFFECTIVE.backup_root), key="p_bak", disabled=True)
    st.text_input("VS_ROOT", str(EFFECTIVE.vs_root), key="p_vs", disabled=True)

PDF_ROOT: Path = EFFECTIVE.pdf_root
VS_ROOT: Path  = EFFECTIVE.vs_root

# --- その他 UI ---
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    backend = st.radio("埋め込みバックエンド", ["openai", "local"], index=0, horizontal=True)
    if backend == "openai":
        st.caption(f"🔧 Embedding モデルは **{OPENAI_EMBED_MODEL}（3072次元）固定**")
with col2:
    chunk_size = st.number_input("チャンクサイズ（文字）", 200, 3000, 900, 50)
    overlap    = st.number_input("オーバーラップ（文字）", 0, 600, 150, 10)
with col3:
    batch_size = st.number_input("埋め込みバッチ数", 8, 512, 64, 8)
    st.caption("※ OCRが必要なPDFは、事前に検索可能PDF化（ocrmypdf 等）しておくと安定します。")

st.info(
    "PDF 入力: `<ssd>/bot_data/pdf/<shard>`（location により自動切替）\n"
    "出力: `./data/vectorstore/<backend>/<shard>/`（バックエンドごとに分離）"
)

# ============================================================
# パスヘルパ（path helpers）
# ============================================================
def list_shards() -> List[str]:
    if not PDF_ROOT.exists():
        return []
    return sorted([p.name for p in PDF_ROOT.iterdir() if p.is_dir()])

def list_pdfs(shard_id: str) -> List[Path]:
    d = PDF_ROOT / shard_id
    if not d.exists():
        return []
    return sorted(d.glob("*.pdf"))

def ensure_vs_dir(backend: str, shard_id: str) -> Path:
    d = VS_ROOT / backend / shard_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_vector_count(base_dir: Path) -> int:
    p = base_dir / "vectors.npy"
    if not p.exists():
        return 0
    try:
        arr = np.load(p, mmap_mode="r")
        return int(arr.shape[0]) if arr.ndim == 2 else 0
    except Exception:
        return 0

# ============================================================
# processed_files.json の canon 化（shard/filename）を保証
# ============================================================
def migrate_processed_files_to_canonical(pf_json: Path, shard_id: str) -> None:
    """
    processed_files.json を 'shard/filename' 形式に正規化（canonicalize）する。
    - 文字列表現: 'file.pdf' → 'shard/file.pdf'
    - dict 表現: file/path/name のいずれかのキーに入っている場合に補完
    - 重複は除去
    """
    pf_list = load_processed_files(pf_json)
    if not pf_list:
        return

    changed = False
    canonical_entries = []

    for entry in pf_list:
        if isinstance(entry, str):
            val = entry
        elif isinstance(entry, dict):
            val = entry.get("file") or entry.get("path") or entry.get("name")
        else:
            continue

        if not val:
            continue

        if "/" not in val:
            val = f"{shard_id}/{val}"
            changed = True

        canonical_entries.append(val)

    # 重複除去
    dedup = []
    seen = set()
    for v in canonical_entries:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)

    if changed:
        save_processed_files(pf_json, dedup)

# ============================================================
# シャード確認
# ============================================================
shards = list_shards()
if not shards:
    st.warning(f"{PDF_ROOT} 配下にサブフォルダー（=シャード=年度）がありません。例: {PDF_ROOT}/2025/*.pdf")
    st.stop()

with st.sidebar:
    st.subheader("対象シャード")
    selected_shards = st.multiselect("複数選択可", shards, default=shards)

st.info("PDF ルート直下の各フォルダーをシャードとして取り込みます。既に取り込んだ PDF はスキップします。")

run = st.button("選択シャード内の PDF を取り込み", type="primary")

# ============================================================
# 実行
# ============================================================
if run:
    # ✅ EmbeddingStore はコンストラクタで openai_model を受け取る（embed() に model= は渡さない）
    estore = EmbeddingStore(backend=backend, openai_model=OPENAI_EMBED_MODEL)
    total_files = 0
    total_chunks = 0

    # 進捗表示を強化：全体用・ファイル用・現在状況テキスト
    overall_progress = st.progress(0.0, text="準備中…")
    file_progress = st.progress(0.0, text="ファイル進捗：待機中…")
    status_current = st.empty()  # 現在シャード / ファイル / ページを逐次表示

    num_shards = len(selected_shards)

    for i_shard, shard_id in enumerate(selected_shards, start=1):
        st.markdown(f"### 📂 シャード: `{shard_id}`")

        try:
            year_val = int(shard_id)
        except ValueError:
            year_val = None

        vs_dir = ensure_vs_dir(backend, shard_id)
        tracker = ProcessedFilesSimple(vs_dir / "processed_files.json")
        vdb = NumpyVectorDB(vs_dir)

        # ✅ 取り込み前に PF を canon 化（旧データ：ファイル名だけ → shard/filename）
        migrate_processed_files_to_canonical(vs_dir / "processed_files.json", shard_id)

        shard_new_files = 0
        shard_new_chunks = 0

        pdf_files = list_pdfs(shard_id)
        n_files = len(pdf_files)
        if n_files == 0:
            st.info("このシャードに PDF がありません。スキップします。")
            overall_progress.progress(i_shard / num_shards, text=f"全体 {i_shard}/{num_shards} シャード完了")
            continue

        for i_file, pdf_path in enumerate(pdf_files, start=1):
            name = pdf_path.name
            key_full = f"{shard_id}/{name}"  # ✅ 正準キー（meta.jsonl と揃える）

            # 互換: 旧キー（name のみ）もスキップ対象に含める
            if tracker.is_done(key_full) or tracker.is_done(name):
                status_current.info(f"⏭️ スキップ: `{shard_id}` / **{name}**（既に取り込み済み）")
                # ファイル進捗は100%にしてから次へ
                file_progress.progress(1.0, text=f"ファイル {i_file}/{n_files} 完了: {name}")
                continue

            try:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    total_pages = max(len(pdf.pages), 1)
                    status_current.info(f"📥 取り込み開始: `{shard_id}` / **{name}**（{i_file}/{n_files}） 全{total_pages}ページ")
                    file_progress.progress(0.0, text=f"ファイル {i_file}/{n_files}: {name} - 0/{total_pages} ページ")

                    for page_no, page in enumerate(pdf.pages, start=1):
                        raw = page.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
                        raw = raw.replace("\t", " ").replace("\xa0", " ")
                        text = " ".join(raw.split())
                        if not text:
                            # 空ページでも進捗だけ更新
                            file_progress.progress(page_no / total_pages, text=f"ファイル {i_file}/{n_files}: {name} - {page_no}/{total_pages} ページ")
                            continue

                        # 正規化 → 分割
                        text = normalize_ja_text(text)
                        spans: List[Tuple[str, int, int]] = split_text(
                            text, chunk_size=int(chunk_size), overlap=int(overlap)
                        )
                        if not spans:
                            file_progress.progress(page_no / total_pages, text=f"ファイル {i_file}/{n_files}: {name} - {page_no}/{total_pages} ページ")
                            continue

                        chunks = [s[0] for s in spans]
                        vectors: List[np.ndarray] = []
                        metas: List[dict] = []

                        for i in range(0, len(chunks), int(batch_size)):
                            batch = chunks[i:i + int(batch_size)]
                            vecs = estore.embed(batch, batch_size=int(batch_size)).astype("float32")
                            vectors.append(vecs)
                            for j, (ch, s, e) in enumerate(spans[i:i + int(batch_size)]):
                                metas.append({
                                    "file": key_full,
                                    "year": year_val,
                                    "page": page_no,
                                    "chunk_id": f"{name}#p{page_no}-{i+j}",
                                    "chunk_index": i + j,
                                    "text": ch,
                                    "span_start": s,
                                    "span_end": e,
                                    "chunk_len_tokens": count_tokens(ch),
                                    "ingested_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                    "shard_id": shard_id,
                                    "embed_model": OPENAI_EMBED_MODEL if backend == "openai" else "local-model",
                                })

                        vec_mat = np.vstack(vectors) if len(vectors) > 1 else vectors[0]
                        vdb.add(vec_mat, metas)
                        shard_new_chunks += len(metas)

                        # ページ到達のたびにファイル進捗更新
                        file_progress.progress(page_no / total_pages, text=f"ファイル {i_file}/{n_files}: {name} - {page_no}/{total_pages} ページ")

                # 取り込み完了を正準キーで記録
                tracker.mark_done(key_full)
                shard_new_files += 1
                status_current.success(f"✅ 完了: `{shard_id}` / **{name}**（{i_file}/{n_files}）")

            except Exception as e:
                st.error(f"❌ 取り込み失敗: {name} : {e}")
                status_current.error(f"❌ 失敗: `{shard_id}` / **{name}** - {e}")

            # シャード内のファイルが1つ終わるごとに全体テキストを更新
            overall_progress.progress(
                (i_shard - 1 + i_file / max(n_files, 1)) / num_shards,
                text=f"全体 {i_shard}/{num_shards} シャード処理中…（{shard_id}: {i_file}/{n_files} ファイル）"
            )

        st.success(f"新規ファイル {shard_new_files} 件 / 追加チャンク {shard_new_chunks} 件")
        st.caption(f"シャード内ベクトル総数（DB計測）: {get_vector_count(vs_dir):,d}")

        # シャード完了時に全体進捗を更新
        overall_progress.progress(i_shard / num_shards, text=f"全体 {i_shard}/{num_shards} シャード完了")

        total_files  += shard_new_files
        total_chunks += shard_new_chunks

    st.toast(f"✅ 完了: 新規 {total_files} ファイル / {total_chunks} チャンク（page + year付き）", icon="✅")

    # ---------- 料金計算（pricing） ----------
    if total_chunks > 0:
        if backend == "openai":
            total_tokens = 0
            # 選択シャードの meta.jsonl から chunk_len_tokens を合算
            for shard_id in selected_shards:
                vs_dir = ensure_vs_dir(backend, shard_id)
                meta_path = vs_dir / "meta.jsonl"
                if not meta_path.exists():
                    continue
                with meta_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        total_tokens += int(obj.get("chunk_len_tokens", 0))

            model = OPENAI_EMBED_MODEL  # large 固定
            usd = pricing.estimate_embedding_cost_usd(total_tokens, model)
            jpy = pricing.estimate_embedding_cost_jpy(total_tokens, model)

            st.markdown("### 💰 埋め込みコストの概算")
            st.write(f"- モデル: **{model}**")
            st.write(f"- 総トークン数: {total_tokens:,}")
            st.write(f"- 概算コスト: `${usd:.4f}` ≈ ¥{jpy:,.0f}")
        else:
            st.markdown("### 💰 埋め込みコストの概算")
            st.info("local backend のためコストは発生しません。")
