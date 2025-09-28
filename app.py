import streamlit as st
from lib.ui import hide_deploy_button
#from dotenv import load_dotenv

#load_dotenv()
st.set_page_config(page_title="社内ボット (No-FAISS版)", page_icon="🤖", layout="wide")
hide_deploy_button()   # ← 最上部で1回呼ぶ

st.title("🤖 社内ボット (no-FAISS版)")
st.markdown("""
左の **Pages** から  
- **pdfベクトル化**：`pdf/` にある .pdf を分割→埋め込み→保存  
- **ボット（改良版）**：保存した知識ベースに対して質問  
を実行します。
""")

st.info("ボット（改良版）を使ってください。右側のサイドメニュー（ボット（改良版））をクリックしてください．")
