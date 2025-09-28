# lib/ui_sidebar.py
import streamlit as st
import os
from pathlib import Path
from lib.openai_utils import is_gpt5

def sidebar_controls(VS_ROOT: Path):
    backend = st.radio("バックエンド", ["openai", "local"], index=0, horizontal=True)
    base_dir = VS_ROOT / backend
    if not base_dir.exists():
        st.error(f"vectorstore/{backend} が見つかりません。")
        st.stop()

    shard_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    shard_ids = [p.name for p in shard_dirs]
    sel_shards = st.multiselect("対象シャード", shard_ids, default=shard_ids)

    st.divider(); st.subheader("絞り込み")
    year_min = st.number_input("年下限", value=0)
    year_max = st.number_input("年上限", value=9999)
    file_filter = st.text_input("ファイル名フィルタ", value="").strip()

    st.divider(); st.subheader("表示設定")
    max_rows = st.number_input("最大表示件数", 50, 5000, 500, 50)
    snippet_len = st.slider("スニペット長", 80, 800, 240, 20)
    show_cols = st.multiselect(
        "表示カラム",
        ["file","year","page","shard_id","chunk_id","chunk_index","score","text"],
        default=["file","year","page","shard_id","score","text"]
    )

    st.divider(); st.subheader("生成オプション（OpenAI）")
    def _get_openai_key():
        try:
            return (
                st.secrets.get("OPENAI_API_KEY")
                or (st.secrets.get("openai") or {}).get("api_key")
                or os.getenv("OPENAI_API_KEY")
            )
        except Exception:
            return os.getenv("OPENAI_API_KEY")

    OPENAI_API_KEY = _get_openai_key()
    has_key = bool(OPENAI_API_KEY)

    gen_enabled = st.checkbox("ヒット要約を生成する", value=has_key, disabled=not has_key)
    model = st.selectbox("モデル",
                         ["gpt-5-mini", "gpt-5", "gpt-4o-mini", "gpt-4o"],
                         index=0, disabled=not gen_enabled)

    if is_gpt5(model):
        temperature = 1.0
        st.metric("temperature", "1.0")
    else:
        temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05, disabled=not gen_enabled)

    max_tokens = st.slider("出力トークン上限", 128, 32000, 2000, 128, disabled=not gen_enabled)
    topn_snippets = st.slider("上位スニペット数", 5, 200, 30, 5, disabled=not gen_enabled)
    auto_batch = st.checkbox("オーバー時は自動バッチ要約", value=True, disabled=not gen_enabled)
    verbose_log = st.checkbox("詳細ログ", value=True, disabled=not gen_enabled)
    debug_mode = st.checkbox("デバッグ情報", value=False, disabled=not gen_enabled)

    sys_prompt = st.text_area("System Prompt",
        "あなたは事実に忠実なリサーチアシスタントです。", disabled=not gen_enabled)
    user_prompt_tpl = st.text_area("User Prompt テンプレート",
        "クエリ『{query}』について以下のスニペットを根拠にまとめてください。\n\n{snippets}",
        disabled=not gen_enabled)

    return (base_dir, sel_shards, year_min, year_max, file_filter,
            max_rows, snippet_len, show_cols,
            gen_enabled, model, temperature, max_tokens,
            topn_snippets, auto_batch, verbose_log, debug_mode,
            sys_prompt, user_prompt_tpl, OPENAI_API_KEY)
