import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from rag_utils import load_txt_files, split_text, EmbeddingStore, NumpyVectorDB, ProcessedFilesSimple

# ====== 初期化 ======
load_dotenv()
st.set_page_config(page_title="Build KB from TXT (Structured Meta)", page_icon="🧱", layout="wide")
st.title("🧱 Build Knowledge Base from `text/*.txt` (Structured Meta)")

text_dir = Path("text")
text_dir.mkdir(exist_ok=True)

# ====== オプション UI ======
with st.expander("📥 オプション設定", expanded=True):
    backend = st.radio("埋め込みバックエンド", ["local", "openai"], index=1)
    chunk_size = st.slider("チャンクサイズ（文字数）", 300, 2000, 900, 50)
    overlap = st.slider("オーバーラップ（文字数）", 0, 400, 150, 10)
    batch_size = st.slider("埋め込みバッチサイズ", 8, 256, 64, 8)
    reset = st.checkbox("既存のベクトルストアを初期化（作り直す）", value=False)
    skip_done = st.checkbox("前回処理済みファイルをスキップ（ファイル名で判定）", value=True)

# ====== 保存先ディレクトリ ======
store_name = "openai" if backend == "openai" else "local"
vs_dir = Path("vectorstore") / store_name
vs_dir.mkdir(parents=True, exist_ok=True)

# ====== 処理済みファイル管理 ======
pstore = ProcessedFilesSimple(vs_dir / "processed_files.json")

# ====== 実行ボタン ======
if st.button("🔨 TXT をベクトル化して保存", type="primary"):
    files = load_txt_files(text_dir)  # -> [(fname, text)]
    if not files:
        st.warning("`text/` に .txt がありません。")
        st.stop()

    if store_name == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY が .env に設定されていません。")
        st.stop()

    estore = EmbeddingStore(backend=store_name)
    vdb = NumpyVectorDB(vs_dir)

    if reset:
        vdb.reset()
        pstore.reset()
        st.info(f"既存データを初期化しました：{vs_dir}")

    to_process = []
    skipped = 0
    for fname, text in files:
        if skip_done and pstore.is_done(fname):
            skipped += 1
            continue
        chs = split_text(text, chunk_size=chunk_size, overlap=overlap)  # [(chunk, start, end)]
        if chs:
            to_process.append((fname, chs))

    if not to_process:
        st.success(f"処理対象はありません（スキップ {skipped} 件）。")
        st.stop()

    total_chunks = sum(len(chs) for _, chs in to_process)
    st.write(f"対象ファイル: **{len(to_process)}** 件 / チャンク総数: **{total_chunks}**")

    prog = st.progress(0.0)
    done_chunks = 0

    for fname, chs in to_process:
        texts = []
        meta_items = []
        for idx, (chunk, start, end) in enumerate(chs):
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

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_meta = meta_items[i:i + batch_size]
            embs = estore.embed(batch_texts, batch_size=batch_size)
            vdb.add(embs, batch_meta)
            done_chunks += len(batch_texts)
            prog.progress(done_chunks / total_chunks)

        pstore.mark_done(fname)

    st.success(f"✅ ベクトル化完了: {total_chunks} チャンクを保存しました ({vs_dir})")
    st.caption("meta.jsonl には file/page/chunk_id/chunk_index/span_start/span_end を含みます。")
