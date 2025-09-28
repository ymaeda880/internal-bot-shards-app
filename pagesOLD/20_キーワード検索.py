# pages/20_キーワード検索.py
# ------------------------------------------------------------
# 🔎 meta.jsonl 横断検索 + （任意）OpenAI 生成要約
# - 二重確認は session_state で保持
# - 失敗時はUIに明示＆ローカル抽出サマリでフォールバック
# - 共通処理は lib/ 以下へ分離
# ------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import streamlit as st

from lib.ui_sidebar import sidebar_controls
from lib.ui_query import query_form
from lib.search_engine import run_search
from lib.summarizer import run_summary
from lib.ui_snippets import render_snippets

# パス
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
VS_ROOT  = DATA_DIR / "vectorstore"

# Streamlit
st.set_page_config(page_title="20 キーワード検索（meta横断）", page_icon="🔎", layout="wide")
st.title("🔎 キーワード検索（meta.jsonl 横断）")

# サイドバー（検索対象・表示設定・生成オプション）
with st.sidebar:
    (
        base_dir, sel_shards, year_min, year_max, file_filter,
        max_rows, snippet_len, show_cols,
        gen_enabled, model, temperature, max_tokens,
        topn_snippets, auto_batch, verbose_log, debug_mode,
        sys_prompt, user_prompt_tpl, OPENAI_API_KEY
    ) = sidebar_controls(VS_ROOT)

# 検索フォーム（メイン）
query, bool_mode, use_regex, case_sensitive, normalize_query, norm_body, go = query_form()

# 検索実行
if go:
    df = run_search(
        sel_shards=sel_shards,
        base_dir=base_dir,
        query=query,
        year_min=year_min,
        year_max=year_max,
        file_filter=file_filter,
        max_rows=max_rows,
        snippet_len=snippet_len,
        normalize_query=normalize_query,
        norm_body=norm_body,
        bool_mode=bool_mode,
        use_regex=use_regex,
        case_sensitive=case_sensitive,
        show_cols=show_cols,
    )

    if df is not None:
        # 一覧とCSV
        st.dataframe(df[[c for c in show_cols if c in df.columns]], use_container_width=True, height=420)
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 CSV ダウンロード", data=csv_bytes, file_name="keyword_hits.csv", mime="text/csv")

        # 要約
        if gen_enabled:
            run_summary(
                df=df, model=model, temperature=temperature, max_tokens=max_tokens,
                topn_snippets=topn_snippets, auto_batch=auto_batch,
                verbose_log=verbose_log, debug_mode=debug_mode,
                sys_prompt=sys_prompt, user_prompt_tpl=user_prompt_tpl,
                OPENAI_API_KEY=OPENAI_API_KEY, query=query
            )

        # スニペット
        if "text" in show_cols:
            render_snippets(df)

else:
    st.info("左のサイドバーで条件を設定し、キーワードを入力して『検索を実行』してください。")
