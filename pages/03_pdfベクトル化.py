# pages/03_pdfベクトル化.py
# ------------------------------------------------------------
# 📥 data/pdf/<year>/*.pdf を取り込み、data/vectorstore/<backend>/<year>/ に
#    vectors.npy / meta.jsonl を “ページ単位” で追記する。
#    meta には year と page を付与。rag_utils.py の API に準拠。
#    重要: meta.jsonl への追記は NumpyVectorDB.add() が行うため、二重追記はしない。
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import unicodedata
import re
import streamlit as st
import pdfplumber
import numpy as np

from rag_utils import split_text, EmbeddingStore, NumpyVectorDB, ProcessedFilesSimple

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
PDF_ROOT = DATA_DIR / "pdf"           # data/pdf/<year>/*.pdf
VS_ROOT  = DATA_DIR / "vectorstore"   # data/vectorstore/<backend>/<year>/

# ---------- helpers ----------
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
    """vectors.npy の行数を返す。存在しなければ 0。"""
    p = base_dir / "vectors.npy"
    if not p.exists():
        return 0
    try:
        arr = np.load(p, mmap_mode="r")
        return int(arr.shape[0]) if arr.ndim == 2 else 0
    except Exception:
        return 0

# ---------- UI ----------
st.set_page_config(page_title="03 ベクトル化（ページ単位・シャード）", page_icon="🧱", layout="wide")
st.title("🧱 フォルダー（=シャード）ごとのベクトル化（page + year付き）")

col1, col2, col3 = st.columns([1,1,2])
with col1:
    backend = st.radio("埋め込みバックエンド", ["openai", "local"], index=0, horizontal=True)
with col2:
    chunk_size = st.number_input("チャンクサイズ（文字）", 200, 3000, 900, 50)
    overlap    = st.number_input("オーバーラップ（文字）", 0, 600, 150, 10)
with col3:
    batch_size = st.number_input("埋め込みバッチ数", 8, 512, 64, 8)
    st.caption("※ OCRが必要なPDFは、事前に検索可能PDF化（ocrmypdf 等）しておくと安定します。")

shards = list_shards()
if not shards:
    st.warning("data/pdf/ 配下にサブフォルダー（=シャード=年度）がありません。例: data/pdf/2025/*.pdf")
    st.stop()

with st.sidebar:
    st.subheader("対象シャード")
    selected_shards = st.multiselect("複数選択可", shards, default=shards)

run = st.button("選択シャードを取り込み（追記）", type="primary")

# ---------- RUN ----------
if run:
    estore = EmbeddingStore(backend=backend)
    total_files = 0
    total_chunks = 0

    progress = st.progress(0.0, text="準備中…")

    for i_shard, shard_id in enumerate(selected_shards, start=1):
        st.markdown(f"### 📂 シャード: `{shard_id}`")

        # フォルダ名を year（数値）に
        try:
            year_val = int(shard_id)
        except ValueError:
            year_val = None

        vs_dir = ensure_vs_dir(backend, shard_id)
        tracker = ProcessedFilesSimple(vs_dir / "processed_files.json")
        vdb = NumpyVectorDB(vs_dir)

        shard_new_files = 0
        shard_new_chunks = 0

        pdf_files = list_pdfs(shard_id)
        if not pdf_files:
            st.info("このシャードにPDFがありません。スキップします。")
            progress.progress(i_shard/len(selected_shards), text=f"{i_shard}/{len(selected_shards)} 完了")
            continue

        for pdf_path in pdf_files:
            name = pdf_path.name
            if tracker.is_done(name):
                continue

            try:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    for page_no, page in enumerate(pdf.pages, start=1):
                        raw = page.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
                        raw = raw.replace("\t", " ").replace("\xa0", " ")
                        text = " ".join(raw.split())
                        if not text:
                            continue

                        # split_text は (chunk, start, end)
                        spans: List[Tuple[str,int,int]] = split_text(
                            text, chunk_size=int(chunk_size), overlap=int(overlap)
                        )
                        if not spans:
                            continue

                        chunks = [s[0] for s in spans]
                        vectors: List[np.ndarray] = []
                        metas: List[dict] = []

                        for i in range(0, len(chunks), int(batch_size)):
                            batch = chunks[i:i+int(batch_size)]
                            vecs = estore.embed(batch, batch_size=int(batch_size)).astype("float32")
                            vectors.append(vecs)
                            for j, (ch, s, e) in enumerate(spans[i:i+int(batch_size)]):
                                metas.append({
                                    "file": f"{shard_id}/{name}",
                                    "year": year_val,
                                    "page": page_no,
                                    "chunk_id": f"{name}#p{page_no}-{i+j}",
                                    "chunk_index": i + j,
                                    "text": normalize_ja_text(ch),   # ★ここで正規化
                                    "span_start": s,
                                    "span_end": e,
                                    "ingested_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                    "shard_id": shard_id
                                })

                        vec_mat = np.vstack(vectors) if len(vectors) > 1 else vectors[0]
                        vdb.add(vec_mat, metas)

                        shard_new_chunks += len(metas)

                tracker.mark_done(name)
                shard_new_files += 1

            except Exception as e:
                st.error(f"❌ 取り込み失敗: {name} : {e}")

        st.success(f"新規ファイル {shard_new_files} 件 / 追加チャンク {shard_new_chunks} 件")
        st.caption(f"シャード内ベクトル総数（DB計測）: {get_vector_count(vs_dir):,d}")

        total_files  += shard_new_files
        total_chunks += shard_new_chunks

        progress.progress(i_shard/len(selected_shards), text=f"{i_shard}/{len(selected_shards)} 完了")

    st.toast(f"✅ 完了: 新規 {total_files} ファイル / {total_chunks} チャンク（page + year付き）", icon="✅")
