# ============================================
# 変更点（この版での修正・追加）
# --------------------------------------------
# 1) 生成オプションを追加（OpenAI）:
#    - サイドバーに「🧠 生成オプション」を追加し、次を設定可能に。
#      ・ヒット要約を生成（有効/無効）
#      ・使用モデル、temperature、max_tokens
#      ・System Prompt、User Prompt テンプレート（{query}, {snippets} を埋め込み）
#      ・生成に使うスニペット件数（Top-N）
# 2) プロンプト微調整可能:
#    - User Prompt（例: 「以下のヒットスニペットを基に『{query}』について要点をまとめて」）を UI から編集可能。
# 3) 実行部に生成フェーズを追加:
#    - 検索結果 DataFrame 作成後、選択された上位 N スニペットを結合して OpenAI に投入。
#    - 生成結果を画面出力（"🧠 生成要約" セクション）。
# 4) 安全性向上:
#    - OPENAI_API_KEY 未設定時は生成オプションを自動的に無効化し警告。
#    - スニペット中の HTML（<mark> 等）を除去してから LLM に渡す。
# ============================================

# pages/04_キーワード検索.py
# ------------------------------------------------------------
# 🔎 キーワード / 正規表現で meta.jsonl（テキスト）を横断検索
# - data/vectorstore/<backend>/<shard_id>/meta.jsonl を読み込み
# - AND/OR, 大文字小文字, 正規表現, 日本語スペース正規化(NFKC + CJK間スペース除去)
# - シャード/年/ファイル絞り込み、結果ハイライト表示、CSV出力、year/file.pdf の📋コピー
# - （オプション）ヒットスニペットを OpenAI に投げて要約を生成（プロンプト微調整可）
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Iterable, Any, Tuple
from datetime import datetime  # noqa
import re
import json
import unicodedata
import os

import numpy as np  # noqa
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ============== パス ==============
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
VS_ROOT  = DATA_DIR / "vectorstore"

# ============== 日本語正規化（クエリ/本文の揺れ対策） ==============
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

# ============== ユーティリティ（生成用） ==============
def strip_html(s: str) -> str:
    """簡易に HTML タグを除去（<mark> 等）"""
    return re.sub(r"<[^>]+>", "", s or "")

def _count_tokens(text: str, model_hint: str = "gpt-4o-mini") -> int:
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model_hint)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / 4))

# ============== JSONL 読み込み ==============
def iter_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                # 壊れた行はスキップ
                continue

# ============== UI 基本 ==============
load_dotenv()
st.set_page_config(page_title="04 キーワード検索（meta横断）", page_icon="🔎", layout="wide")
st.title("🔎 キーワード検索（meta.jsonl 横断）")

with st.sidebar:
    st.header("検索対象")
    backend = st.radio("バックエンド", ["openai", "local"], index=0, horizontal=True)
    base_dir = VS_ROOT / backend
    if not base_dir.exists():
        st.error(f"vectorstore/{backend} が見つかりません。先に 03 ベクトル化を実行してください。")
        st.stop()

    shard_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    shard_ids = [p.name for p in shard_dirs]
    sel_shards = st.multiselect("対象シャード", shard_ids, default=shard_ids)

    st.divider()
    st.subheader("絞り込み（任意）")
    year_min = st.number_input("年（下限）", value=0, step=1, help="0 で無効")
    year_max = st.number_input("年（上限）", value=9999, step=1, help="9999 で無効")
    file_filter = st.text_input("ファイル名フィルタ（部分一致 / 例: budget）", value="").strip()

    st.divider()
    st.subheader("表示設定")
    max_rows = st.number_input("最大表示件数", min_value=50, max_value=5000, value=500, step=50)
    snippet_len = st.slider("スニペット長（前後合計）", min_value=80, max_value=800, value=240, step=20)
    show_cols = st.multiselect(
        "表示カラム",
        ["file","year","page","shard_id","chunk_id","chunk_index","score","text"],
        default=["file","year","page","shard_id","score","text"]
    )

    st.divider()
    st.subheader("🧠 生成オプション（OpenAI）")
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    gen_enabled = st.checkbox("ヒット要約を生成する", value=False, disabled=not has_key)
    if not has_key:
        st.warning("OPENAI_API_KEY が未設定のため、生成は無効です。", icon="⚠️")
    model = st.selectbox("モデル", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0, disabled=not gen_enabled)
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05, disabled=not gen_enabled)
    max_tokens = st.slider("max_tokens", 128, 4000, 1000, 64, disabled=not gen_enabled)
    topn_snippets = st.slider("生成に使う上位スニペット数", 5, 200, 30, 5, disabled=not gen_enabled)

    sys_prompt = st.text_area(
        "System Prompt",
        value="あなたは事実に忠実なリサーチアシスタントです。根拠のある記述のみを日本語で簡潔にまとめてください。",
        height=80,
        disabled=not gen_enabled
    )
    user_prompt_tpl = st.text_area(
        "User Prompt テンプレート（{query}, {snippets} を埋め込み）",
        value=(
            "以下はキーワード検索で得られたヒットスニペットです。"
            "この情報【のみ】を根拠に、クエリ『{query}』について要点を箇条書き→短いまとめの順で整理してください。"
            "\n\n# ヒットスニペット\n{snippets}"
        ),
        height=140,
        disabled=not gen_enabled
    )

# ============== 検索フォーム ==============
st.markdown("### クエリ")
c1, c2 = st.columns([3,2])
with c1:
    query = st.text_input("キーワード（空白区切りで AND / OR 指定）", value="")
with c2:
    bool_mode = st.radio("モード", ["AND", "OR"], index=0, horizontal=True)

c3, c4, c5, c6 = st.columns(4)
with c3:
    use_regex = st.checkbox("正規表現", value=False)
with c4:
    case_sensitive = st.checkbox("大文字小文字を区別", value=False)
with c5:
    normalize_query = st.checkbox("日本語スペース正規化（推奨）", value=True)
with c6:
    norm_body = st.checkbox("本文も正規化して検索", value=True, help="取り込み時に正規化していないコーパス向け")

go = st.button("検索を実行", type="primary")

# ============== 検索ロジック ==============
def to_flags(case_sensitive: bool) -> int:
    return 0 if case_sensitive else re.IGNORECASE

def compile_terms(q: str, use_regex: bool, case_sensitive: bool) -> List[re.Pattern]:
    if normalize_query:
        q = normalize_ja_text(q)
    terms = [t for t in q.split() if t]
    if not terms:
        return []
    flags = to_flags(case_sensitive)
    pats = []
    for t in terms:
        if use_regex:
            try:
                pats.append(re.compile(t, flags))
            except re.error:
                # 不正な正規表現はリテラル扱い
                pats.append(re.compile(re.escape(t), flags))
        else:
            pats.append(re.compile(re.escape(t), flags))
    return pats

def find_first_span(text: str, pats: List[re.Pattern]) -> Tuple[int,int,List[str]]:
    """
    最初に見つかったヒット位置（min start, max end）と、ヒットした語の一覧を返す
    """
    hits = []
    s_min = None
    e_max = None
    for p in pats:
        m = p.search(text)
        if m:
            hits.append(p.pattern)
            s, e = m.start(), m.end()
            s_min = s if s_min is None else min(s_min, s)
            e_max = e if e_max is None else max(e_max, e)
    if s_min is None:
        return -1, -1, []
    return s_min, e_max, hits

def make_snippet(text: str, pats: List[re.Pattern], total_len: int = 240) -> str:
    s, e, _ = find_first_span(text, pats)
    if s < 0:
        s, e = 0, min(len(text), total_len)
    margin = max(0, total_len // 2)
    left = max(0, s - margin)
    right = min(len(text), e + margin)
    snippet = text[left:right]

    # ハイライト（HTML）
    for p in pats:
        try:
            snippet = p.sub(lambda m: f"<mark>{m.group(0)}</mark>", snippet)
        except re.error:
            pass
    # 端に省略記号
    if left > 0:
        snippet = "…"+snippet
    if right < len(text):
        snippet = snippet+"…"
    return snippet

def copy_button(text: str, label: str, key: str):
    payload = json.dumps(text, ensure_ascii=False)
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
            console.error(e);
            alert("コピーに失敗しました: " + e);
          }}
        }});
      }}
    </script>
    """
    st.components.v1.html(html, height=38)

if go:
    if not sel_shards:
        st.warning("少なくとも1つのシャードを選択してください。")
        st.stop()

    pats = compile_terms(query, use_regex=use_regex, case_sensitive=case_sensitive)
    if not pats:
        st.warning("検索語が空です。キーワードを入力してください。")
        st.stop()

    rows: List[Dict[str,Any]] = []
    total_scanned = 0

    for sid in sel_shards:
        meta_path = base_dir / sid / "meta.jsonl"
        for obj in iter_jsonl(meta_path):
            total_scanned += 1
            # 年・ファイルフィルタ
            yr = obj.get("year", None)
            if isinstance(yr, int):
                if year_min and yr < year_min: continue
                if year_max and year_max < 9999 and yr > year_max: continue
            if file_filter:
                if file_filter.lower() not in str(obj.get("file","")).lower():
                    continue

            text = str(obj.get("text",""))
            if norm_body:
                text_for_match = normalize_ja_text(text)
            else:
                text_for_match = text

            # マッチ判定
            if bool_mode == "AND":
                ok = all(p.search(text_for_match) for p in pats)
            else:
                ok = any(p.search(text_for_match) for p in pats)

            if not ok:
                continue

            # スコア = マッチ語の合計出現数（簡易）
            score = 0
            for p in pats:
                score += len(list(p.finditer(text_for_match)))

            rows.append({
                "file": obj.get("file"),
                "year": obj.get("year"),
                "page": obj.get("page"),
                "shard_id": obj.get("shard_id", sid),
                "chunk_id": obj.get("chunk_id"),
                "chunk_index": obj.get("chunk_index"),
                "score": int(score),
                "text": make_snippet(text, pats, total_len=int(snippet_len)),
            })

            if len(rows) >= int(max_rows):
                break
        if len(rows) >= int(max_rows):
            break

    if not rows:
        st.warning("ヒットなし。検索語やフィルタを調整してください。")
        st.stop()

    # スコア降順で並べ替え
    df = pd.DataFrame(rows).sort_values(["score","year","file","page"], ascending=[False, True, True, True])

    st.success(f"ヒット {len(df):,d} 件 / 走査 {total_scanned:,d} レコード（上位のみ表示）")
    # HTMLハイライトを効かせるため text 列は markdown 表示に
    show_order = [c for c in show_cols if c in df.columns]
    if "text" in show_order:
        non_text_cols = [c for c in show_order if c != "text"]
    else:
        non_text_cols = show_order

    st.dataframe(df[non_text_cols], use_container_width=True, height=420)
    if "text" in show_order:
        st.markdown("#### ヒットスニペット")
        for i, row in df.head(200).iterrows():  # スニペット部分は別レンダリング
            colA, colB = st.columns([4,1])
            with colA:
                st.markdown(f"**{row.get('file')}**  year={row.get('year')}  p.{row.get('page')}  "
                            f"score={row.get('score')}", help=row.get("chunk_id"))
                st.markdown(row.get("text",""), unsafe_allow_html=True)
            with colB:
                copy_button(text=str(row.get("file")), label="year/file をコピー", key=f"cpy_{i}")

    # ダウンロード（CSV）
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 CSV をダウンロード", data=csv_bytes, file_name="keyword_hits.csv", mime="text/csv")

    # ============== 生成フェーズ（オプション） ==============
    if gen_enabled:
        try:
            st.divider()
            st.subheader("🧠 生成要約（OpenAI）")

            # 上位 N 件のスニペットを結合（HTML除去）
            take_n = int(topn_snippets)
            selected = df.head(take_n).copy()
            # スニペット本文（text列はHTMLハイライトなので除去）
            joined_snippets = []
            for _, r in selected.iterrows():
                src = f"{r.get('file')} p.{r.get('page')} (score={r.get('score')})"
                snip = strip_html(str(r.get("text","")))
                joined_snippets.append(f"---\n# Source: {src}\n{snip}")
            snippets_text = "\n\n".join(joined_snippets)

            # プロンプト生成
            user_prompt = user_prompt_tpl.format(query=query, snippets=snippets_text)

            # ざっくりトークン数（目安）
            approx_tokens = _count_tokens(user_prompt, model_hint=model)
            st.caption(f"（プロンプト推定トークン: ~{approx_tokens:,} tok）")

            client = OpenAI()
            with st.spinner("OpenAI で要約を生成中…"):
                resp = client.chat.completions.create(
                    model=model,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    messages=[
                        {"role": "system", "content": sys_prompt.strip()},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            out_text = resp.choices[0].message.content
            st.markdown(out_text)

        except Exception as e:
            st.error(f"生成に失敗しました: {e}")

else:
    st.info("左でシャードと条件を選び、キーワードを入力して『検索を実行』してください。")
