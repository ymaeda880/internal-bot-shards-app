# lib/ui_snippets.py
from __future__ import annotations
import json
import streamlit as st
import pandas as pd

def _copy_button(text: str, label: str, key: str):
    """クリップボードに text をコピーする軽量ボタン（純JS）。"""
    payload = json.dumps(text or "", ensure_ascii=False)
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
            alert("コピーに失敗しました: " + e);
          }}
        }});
      }}
    </script>
    """
    st.components.v1.html(html, height=38)

def render_snippets(df: pd.DataFrame, max_show: int = 200):
    """
    検索ヒットのスニペットを折りたたみUIで表示。
    df: columns 例 = ['file','year','page','score','text', ...]
    """
    st.divider()
    with st.expander("ヒットスニペット（クリックで展開）", expanded=False):
        for i, row in df.head(max_show).iterrows():
            colA, colB = st.columns([4, 1])
            with colA:
                file_ = row.get("file")
                year_ = row.get("year")
                page_ = row.get("page")
                score_ = row.get("score")
                chunk_id = row.get("chunk_id")
                st.markdown(
                    f"**{file_}**  year={year_}  p.{page_}  score={score_}",
                    help=str(chunk_id) if chunk_id is not None else None,
                )
                st.markdown(str(row.get("text","")), unsafe_allow_html=True)
            with colB:
                _copy_button(text=str(file_), label="year/file をコピー", key=f"cpy_{i}")
