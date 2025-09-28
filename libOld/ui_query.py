# lib/ui_query.py
import streamlit as st

def query_form():
    st.markdown("### クエリ")
    c1, c2 = st.columns([3,2])
    with c1:
        query = st.text_input("キーワード", value="")
    with c2:
        bool_mode = st.radio("モード", ["AND", "OR"], index=0, horizontal=True)

    c3, c4, c5, c6 = st.columns(4)
    with c3:
        use_regex = st.checkbox("正規表現", value=False)
    with c4:
        case_sensitive = st.checkbox("大文字小文字区別", value=False)
    with c5:
        normalize_query = st.checkbox("日本語スペース正規化", value=True)
    with c6:
        norm_body = st.checkbox("本文も正規化", value=True)

    go = st.button("検索を実行", type="primary")
    return query, bool_mode, use_regex, case_sensitive, normalize_query, norm_body, go
