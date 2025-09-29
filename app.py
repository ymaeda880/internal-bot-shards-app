import streamlit as st
from lib.ui import hide_deploy_button
from config.path_config import PATHS  # ← 追加

st.set_page_config(page_title="社内ボット (No-FAISS版)", page_icon="🤖", layout="wide")
hide_deploy_button()

st.title("🤖 社内ボット (no-FAISS版)")
st.markdown("""
左の **Pages** から  
- **pdfベクトル化**：`pdf/` にある .pdf を分割→埋め込み→保存  
- **ボット（改良版）**：保存した知識ベースに対して質問  
を実行します。
""")

st.info("ボット（改良版）を使ってください。右側のサイドメニュー（ボット（改良版））をクリックしてください．")

# === ここから追加 ===
st.divider()
st.subheader("📂 現在の環境設定")

st.text(f"現在の location: {PATHS.preset}")
st.text(f"APP_ROOT       : {PATHS.app_root}")
st.text(f"pdf_root       : {PATHS.pdf_root}")
st.text(f"backup_root    : {PATHS.backup_root}")
st.text(f"vs_root        : {PATHS.vs_root}")
st.text(f"ssd_path       : {PATHS.ssd_path}")
# === ここまで追加 ===
