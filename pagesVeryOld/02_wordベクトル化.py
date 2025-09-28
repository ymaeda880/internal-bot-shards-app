# pages/Wordベクトル化.py
# ------------------------------------------------------------
# 📝 Build Knowledge Base from Word (.docx) with Structured Meta
# - word/*.docx を読み込み → テキスト抽出（段落＋表）
# - split_text でチャンク化（span_start / span_end を付与）
# - Embedding → vectors.npy 追加保存
# - meta.jsonl に構造化メタを1行1チャンクで追記
#   {file, page=1, chunk_id, chunk_index, text, span_start, span_end}
# - 既存PDF/TXTページと同じUX（reset / skip_done / 進捗バー等）
# ------------------------------------------------------------

import os
from pathlib import Path
from typing import List, Tuple, Dict

import streamlit as st
from dotenv import load_dotenv

from lib.rag_utils import split_text, EmbeddingStore, NumpyVectorDB, ProcessedFilesSimple

# ====== 初期化 ======
load_dotenv()
st.set_page_config(page_title="Build KB from Word (Structured Meta)", page_icon="📝", layout="wide")
st.title("📝 Build Knowledge Base from Word (.docx) (Structured Meta)")

word_dir = Path("word")
word_dir.mkdir(exist_ok=True)

# ====== オプション UI ======
with st.expander("📥 オプション設定", expanded=True):
    backend = st.radio("埋め込みバックエンド", ["local", "openai"], index=1)
    chunk_size = st.slider("チャンクサイズ（文字数）", 300, 2000, 900, 50)
    overlap   = st.slider("オーバーラップ（文字数）", 0, 400, 150, 10)
    batch_size = st.slider("埋め込みバッチサイズ", 8, 256, 64, 8)
    reset = st.checkbox("既存のベクトルストアを初期化（作り直す）", value=False)
    skip_done = st.checkbox("前回処理済みファイルをスキップ（ファイル名で判定）", value=True)

# ====== 保存先ディレクトリ ======
store_name = "openai" if backend == "openai" else "local"
vs_dir = Path("vectorstore") / store_name
vs_dir.mkdir(parents=True, exist_ok=True)

# ====== 処理済みファイル管理 ======
pstore = ProcessedFilesSimple(vs_dir / "processed_files.json")

# ====== Word ローダ ======
def _import_docx():
    try:
        from docx import Document  # python-docx
        return Document
    except Exception as e:
        return None

def load_docx_files(data_dir: Path) -> List[Tuple[str, str]]:
    """
    data_dir 配下の .docx を走査し、(ファイル名, テキスト全文) のリストを返す。
    - 段落に加えて、表（tables）内のセル文字列も抽出
    - 余分な空行を詰める
    """
    Document = _import_docx()
    if Document is None:
        raise RuntimeError("python-docx が見つかりません。`pip install python-docx` を実行してください。")

    results: List[Tuple[str, str]] = []
    for p in sorted(data_dir.glob("*.docx")):
        try:
            doc = Document(p)
            texts: List[str] = []

            # 段落
            for para in doc.paragraphs:
                t = (para.text or "").strip()
                if t:
                    texts.append(t)

            # 表（行×列）も文字を抽出
            for tbl in getattr(doc, "tables", []):
                for row in tbl.rows:
                    row_cells = []
                    for cell in row.cells:
                        ct = (cell.text or "").strip()
                        row_cells.append(ct)
                    # タブ区切りで1行に
                    if any(row_cells):
                        texts.append("\t".join(row_cells))

            # 結合 & 空行詰め
            raw = "\n".join(texts)
            # 連続改行の圧縮
            cleaned = "\n".join([line for line in raw.splitlines() if line.strip() != ""]).strip()

            if cleaned:
                results.append((p.name, cleaned))
        except Exception as e:
            st.warning(f"Word 読み込み失敗: {p.name} / {e}")

    return results

# ====== 実行ボタン ======
st.write("📂 `word/` に .docx を置いてから実行してください。")
if st.button("🔨 Word をベクトル化して保存", type="primary"):
    try:
        files = load_docx_files(word_dir)  # -> [(fname, full_text)]
    except Exception as e:
        st.error(f"Word 読み込み時にエラー: {e}")
        st.stop()

    if not files:
        st.warning("抽出できたテキストがありませんでした。")
        st.stop()

    if store_name == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY が .env に設定されていません。")
        st.stop()

    estore = EmbeddingStore(backend=store_name)
    vdb = NumpyVectorDB(vs_dir)

    if reset:
        vdb.reset()
        pstore.reset()
        st.info(f"初期化しました: {vs_dir}")

    # ====== 前処理（スキップ反映 & チャンク数集計） ======
    to_process: List[Tuple[str, List[Tuple[str, int, int]]]] = []  # (fname, [(chunk, start, end)])
    skipped = 0
    for fname, full_text in files:
        if skip_done and pstore.is_done(fname):
            skipped += 1
            continue
        chs = split_text(full_text, chunk_size=chunk_size, overlap=overlap)  # -> [(chunk, start, end)]
        if chs:
            to_process.append((fname, chs))

    if not to_process:
        st.success(f"処理対象はありません（スキップ {skipped} 件）。")
        st.stop()

    total_chunks = sum(len(chs) for _, chs in to_process)
    st.write(f"対象Word: **{len(to_process)}** 件 / スキップ: **{skipped}** 件 / チャンク総数: **{total_chunks}**")

    prog = st.progress(0.0)
    done_chunks = 0

    # ====== ベクトル化・保存 ======
    try:
        processed_files = 0

        for fname, chs in to_process:
            texts: List[str] = []
            meta_items: List[Dict] = []

            for idx, (chunk, start, end) in enumerate(chs):
                # Word はページ概念が取りにくいので page=1 固定（TXT と同様）
                chunk_id = f"{fname}#p1#{idx}"
                meta_items.append({
                    "file": fname,
                    "page": 1,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "text": chunk,
                    "span_start": start,
                    "span_end": end,
                })
                texts.append(chunk)

            # バッチで埋め込み→保存
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_meta = meta_items[i:i + batch_size]
                embs = estore.embed(batch_texts, batch_size=batch_size)
                vdb.add(embs, batch_meta)
                done_chunks += len(batch_texts)
                prog.progress(done_chunks / total_chunks)

            pstore.mark_done(fname)
            processed_files += 1

        st.success(f"✅ ベクトル化完了: {total_chunks} チャンクを保存しました ({vs_dir})")
        st.caption("meta.jsonl には file/page/chunk_id/chunk_index/span_start/span_end を含みます。")
        st.caption(f"追加 {processed_files} 件 / スキップ {skipped} 件 / チャンク {total_chunks} 件")

    except Exception as e:
        st.error(f"埋め込み中にエラー: {e}")
