# pages/20_キーワード検索.py
# ------------------------------------------------------------
# 🔎 meta.jsonl 横断検索 + （任意）OpenAI 生成要約（自動・スピナー付き）
# - 要約は検索結果表示と同じ実行内で自動生成（ボタンなし）
# - 生成中はスピナー表示（OpenAI実行／ローカル抽出）
# - 失敗/未設定時はローカル抽出サマリを必ず表示（空振りゼロ）
# - フォーム/二重確認/ rerun / stop を使用せず“戻る”問題を根絶（検索のガード以外）
# - 既存の data/vectorstore/{openai,local} 配下の meta.jsonl を走査
# ------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Iterable, Any
import re, json, os, unicodedata, traceback

import pandas as pd
import streamlit as st

# （任意）OpenAI：キーが無くても動きます（ローカル抽出サマリに自動フォールバック）
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# ============== パス ==============
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
VS_ROOT  = DATA_DIR / "vectorstore"

# ============== 基本UI ==============
st.set_page_config(page_title="20 キーワード検索（meta横断）", page_icon="🔎", layout="wide")
st.title("🔎 キーワード検索（meta.jsonl 横断）")

# ============== 日本語正規化 & テキストユーティリティ ==============
CJK = r"\u3040-\u309F\u30A0-\u30FF\uFF65-\uFF9F\u4E00-\u9FFF\u3400-\u4DBF"
PUNC = r"、。・，．！？：；（）［］｛｝「」『』〈〉《》【】"
_cjk_cjk_space = re.compile(fr"(?<=[{CJK}])\s+(?=[{CJK}])")
_space_before_punc = re.compile(fr"\s+(?=[{PUNC}])")
_space_after_open = re.compile(fr"(?<=[（［｛「『〈《【])\s+")
_space_before_close = re.compile(fr"\s+(?=[）］｝」』〉》】])")
_multi_space = re.compile(r"[ \t\u3000]+")

def normalize_ja_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = _cjk_cjk_space.sub("", s)
    s = _space_before_punc.sub("", s)
    s = _space_after_open.sub("", s)
    s = _space_before_close.sub("", s)
    s = _multi_space.sub(" ", s)
    return s.strip()

def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")

def make_snippet(text: str, pats: List[re.Pattern], total_len: int = 240) -> str:
    s_pos, e_pos = 0, 0
    for p in pats:
        m = p.search(text)
        if m:
            s_pos, e_pos = m.start(), m.end()
            break
    if e_pos == 0:
        s_pos, e_pos = 0, min(len(text), total_len)
    margin = total_len // 2
    left = max(0, s_pos - margin)
    right = min(len(text), e_pos + margin)
    snippet = text[left:right]
    for p in pats:
        try:
            snippet = p.sub(lambda m: f"<mark>{m.group(0)}</mark>", snippet)
        except re.error:
            pass
    if left > 0:
        snippet = "…" + snippet
    if right < len(text):
        snippet = snippet + "…"
    return snippet

# ============== トークン見積り（tiktokenが無ければ概算） ==============
def _encoding_for(model_hint: str):
    try:
        import tiktoken
        try:
            return tiktoken.encoding_for_model(model_hint)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def count_tokens(text: str, model_hint: str = "gpt-5-mini") -> int:
    enc = _encoding_for(model_hint)
    if enc is None:
        return max(1, int(len(text or "") / 4))
    try:
        return len(enc.encode(text or ""))
    except Exception:
        return max(1, int(len(text or "") / 4))

def truncate_by_tokens(text: str, max_tokens: int, model_hint: str = "gpt-5-mini") -> str:
    enc = _encoding_for(model_hint)
    if enc is None:
        max_chars = max(100, max_tokens * 4)
        return (text or "")[:max_chars]
    try:
        toks = enc.encode(text or "")
        if len(toks) <= max_tokens:
            return text or ""
        return enc.decode(toks[:max_tokens])
    except Exception:
        max_chars = max(100, max_tokens * 4)
        return (text or "")[:max_chars]

def is_gpt5(model_name: str) -> bool:
    return (model_name or "").lower().startswith("gpt-5")

# ============== JSONL イテレータ ==============
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
                continue

# ============== サイドバー ==============
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
    file_filter = st.text_input("ファイル名フィルタ（部分一致）", value="").strip()

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
    st.subheader("🧠 生成オプション")
    # secrets.toml 優先、なければ環境変数
    def _get_openai_key() -> str | None:
        try:
            return (
                st.secrets.get("OPENAI_API_KEY")
                or (st.secrets.get("openai") or {}).get("api_key")
                or os.getenv("OPENAI_API_KEY")
            )
        except Exception:
            return os.getenv("OPENAI_API_KEY")
    OPENAI_API_KEY = _get_openai_key()

    gen_enabled = st.checkbox("ヒット要約を自動生成する", value=True)
    model = st.selectbox("モデル", ["gpt-5-mini", "gpt-5", "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temperature = 1.0 if is_gpt5(model) else st.slider("temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("出力トークン上限", 128, 32000, 2000, 128)
    topn_snippets = st.slider("要約に使う上位スニペット数", 5, 200, 30, 5)

    sys_prompt = st.text_area(
        "System Prompt",
        value="あなたは事実に忠実なリサーチアシスタントです。根拠のある記述のみを日本語で簡潔にまとめてください。",
        height=80,
    )
    user_prompt_tpl = st.text_area(
        "User Prompt テンプレート（{query}, {snippets} を埋め込み）",
        value=(
            "以下はキーワード検索で得られたヒットスニペットです。"
            "この情報【のみ】を根拠に、クエリ『{query}』について要点を箇条書き→短いまとめの順で整理してください。"
            "\n\n# ヒットスニペット\n{snippets}"
        ),
        height=140,
    )
    debug_mode = st.checkbox("デバッグ情報を表示", value=False)

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
def compile_terms(q: str, use_regex: bool, case_sensitive: bool) -> List[re.Pattern]:
    if normalize_query:
        q = normalize_ja_text(q)
    terms = [t for t in q.split() if t]
    if not terms:
        return []
    flags = 0 if case_sensitive else re.IGNORECASE
    pats = []
    for t in terms:
        if use_regex:
            try:
                pats.append(re.compile(t, flags))
            except re.error:
                pats.append(re.compile(re.escape(t), flags))
        else:
            pats.append(re.compile(re.escape(t), flags))
    return pats

def _local_summary(labelled_snips: List[str], max_sent: int = 10) -> str:
    text = "\n\n".join(labelled_snips)
    text = re.sub(r"(?m)^---\s*$", "", text)
    text = re.sub(r"(?m)^#\s*Source:.*$", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    parts = re.split(r"[。．.!?！？]\s*|\n+", text)
    uniq, out = set(), []
    for p in parts:
        p = (p or "").strip()
        if len(p) < 6 or p in uniq:
            continue
        uniq.add(p)
        out.append(f"・{p}")
        if len(out) >= max_sent:
            break
    if not out:
        return "（ローカル抽出サマリ：要約できる文が見つかりませんでした）"
    short_base = parts[0] if parts else ""
    short = short_base[:120] + ("…" if len(short_base) > 120 else "")
    return "### （ローカル抽出サマリ）\n" + "\n".join(out) + f"\n\n— 短いまとめ: {short}"

def _fit_to_budget(snips: List[str], *, model: str, sys_prompt: str, user_prefix: str,
                   want_output: int, context_limit: int, safety_margin: int) -> List[str]:
    while True:
        toks = count_tokens(sys_prompt, model) + count_tokens(user_prefix, model)
        toks += sum(count_tokens(s, model) for s in snips)
        need = toks + want_output + safety_margin
        if need <= context_limit or not snips:
            break
        snips = snips[:-1]
    if snips:
        budget = context_limit - (count_tokens(sys_prompt, model) + count_tokens(user_prefix, model) + want_output + safety_margin)
        budget = max(500, budget)
        snips = [s if count_tokens(s, model) <= budget else truncate_by_tokens(s, budget, model) for s in snips]
    return snips

def _openai_summary(*, model: str, temperature: float, max_tokens: int,
                    sys_prompt: str, user_prompt: str, api_key: str) -> str:
    if not (_HAS_OPENAI and api_key):
        raise RuntimeError("OpenAI未設定")
    client = OpenAI(api_key=api_key)
    if is_gpt5(model):
        resp = client.responses.create(
            model=model,
            input=[{"role": "system", "content": sys_prompt.strip()},
                   {"role": "user", "content": user_prompt}],
            temperature=float(temperature),
            max_output_tokens=int(max_tokens),
        )
        try:
            return resp.output_text or ""
        except Exception:
            out = getattr(resp, "output", None)
            if out:
                for item in out:
                    if getattr(item, "type", "") == "message":
                        for c in getattr(item, "content", []) or []:
                            t = getattr(c, "text", None)
                            if isinstance(t, str) and t.strip():
                                return t
                    t = getattr(item, "text", None)
                    if isinstance(t, str) and t.strip():
                        return t
            return ""
    else:
        resp = client.chat.completions.create(
            model=model,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            messages=[
                {"role": "system", "content": sys_prompt.strip()},
                {"role": "user", "content": user_prompt},
            ],
        )
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return ""

# ============== 実行 ==============
if go:
    try:
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
            if not meta_path.exists():
                st.warning(f"{meta_path} が見つかりません。スキップします。")
                continue
            for obj in iter_jsonl(meta_path):
                total_scanned += 1
                yr = obj.get("year", None)
                if isinstance(yr, int):
                    if year_min and yr < year_min: continue
                    if year_max and year_max < 9999 and yr > year_max: continue
                if file_filter and file_filter.lower() not in str(obj.get("file","")).lower():
                    continue

                text = str(obj.get("text",""))
                text_for_match = normalize_ja_text(text) if norm_body else text

                ok = all(p.search(text_for_match) for p in pats) if bool_mode == "AND" \
                     else any(p.search(text_for_match) for p in pats)
                if not ok:
                    continue

                score = sum(len(list(p.finditer(text_for_match))) for p in pats)
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

        df = pd.DataFrame(rows).sort_values(["score","year","file","page"], ascending=[False, True, True, True])

        st.success(f"ヒット {len(df):,d} 件 / 走査 {total_scanned:,d} レコード（上位のみ表示）")

        show_order = [c for c in show_cols if c in df.columns]
        if not show_order:
            show_order = ["file","year","page","shard_id","score","text"]
        non_text_cols = [c for c in show_order if c != "text"]
        st.dataframe(df[non_text_cols], use_container_width=True, height=420)

        csv_bytes = df[show_order].to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 CSV をダウンロード", data=csv_bytes, file_name="keyword_hits.csv", mime="text/csv")

        # ============== 🧠 自動要約（スピナー付き） ==============
        if gen_enabled:
            st.divider()
            st.subheader("🧠 生成要約（自動）")

            # 上位 N スニペットを整形
            take_n = int(topn_snippets)
            selected = df.head(take_n).copy()
            labelled_snips: List[str] = []
            for _, r in selected.iterrows():
                src = f"{r.get('file')} p.{r.get('page')} (score={r.get('score')})"
                snip = strip_html(str(r.get("text","")))
                labelled_snips.append(f"---\n# Source: {src}\n{snip}")

            # 1回投げに収める（超過時は末尾から減らす/切り詰め）
            model_hint = model
            context_limit, safety_margin = (128_000, 2_000) if is_gpt5(model) else (128_000, 1_000)
            user_prefix = user_prompt_tpl.replace("{snippets}", "").format(query=query, snippets="")
            fitted = _fit_to_budget(
                labelled_snips, model=model_hint, sys_prompt=sys_prompt, user_prefix=user_prefix,
                want_output=int(max_tokens), context_limit=context_limit, safety_margin=safety_margin
            )
            if not fitted:
                with st.spinner("🧩 ローカル抽出サマリを生成中…"):
                    st.info("⚠️ 入力が大きすぎるため、ローカル抽出サマリを表示します。")
                    st.markdown(_local_summary(labelled_snips, max_sent=12))
            else:
                snippets_text = "\n\n".join(fitted)
                user_prompt = user_prompt_tpl.format(query=query, snippets=snippets_text)
                approx = count_tokens(user_prompt, model_hint) + count_tokens(sys_prompt, model_hint)
                st.caption(f"（推定入力 ~{approx:,} tok / 出力上限 {int(max_tokens):,} tok / コンテキスト~{context_limit:,} tok）")

                try:
                    with st.spinner("🧠 要約を生成中…"):
                        out = _openai_summary(
                            model=model, temperature=float(temperature), max_tokens=int(max_tokens),
                            sys_prompt=sys_prompt, user_prompt=user_prompt, api_key=OPENAI_API_KEY
                        )
                    if not out or not str(out).strip():
                        with st.spinner("🧩 ローカル抽出サマリを生成中…"):
                            st.info("⚠️ モデル出力が空だったため、ローカル抽出サマリを表示します。")
                            st.markdown(_local_summary(fitted, max_sent=12))
                    else:
                        st.markdown(str(out).strip())
                except Exception as e:
                    if debug_mode:
                        st.error(f"OpenAI エラー: {type(e).__name__}: {e}", icon="🛑")
                        st.code("".join(traceback.format_exc()))
                    with st.spinner("🧩 ローカル抽出サマリを生成中…"):
                        st.markdown(_local_summary(fitted, max_sent=12))

        # ============== ヒットスニペット（既定で畳む） ==============
        if "text" in show_order:
            st.divider()
            with st.expander("ヒットスニペット（クリックで展開）", expanded=False):
                for i, row in df.head(200).iterrows():
                    colA, colB = st.columns([4,1])
                    with colA:
                        st.markdown(
                            f"**{row.get('file')}**  year={row.get('year')}  p.{row.get('page')}  "
                            f"score={row.get('score')}",
                            help=row.get("chunk_id")
                        )
                        st.markdown(row.get("text",""), unsafe_allow_html=True)
                    with colB:
                        # ちょっとしたコピー機能（JS）— 失敗しても動作に影響しない
                        payload = json.dumps(str(row.get("file")), ensure_ascii=False)
                        html = f"""
                        <button id="cpy_{i}" style="
                            padding:6px 10px;border-radius:8px;border:1px solid #dadce0;
                            background:#fff;cursor:pointer;font-size:0.9rem;">📋 year/file をコピー</button>
                        <script>
                          const b=document.getElementById("cpy_{i}");
                          if(b){{b.addEventListener("click",async()=>{{
                            try{{await navigator.clipboard.writeText({payload});
                              const o=b.innerText;b.innerText="✅ コピーしました";
                              setTimeout(()=>{{b.innerText=o}},1200);
                            }}catch(e){{alert("コピーに失敗: "+e)}}
                          }})}}
                        </script>
                        """
                        st.components.v1.html(html, height=38)

    except Exception:
        st.error("検索処理でエラーが発生しました。", icon="🛑")
        if debug_mode:
            st.code("".join(traceback.format_exc()))

else:
    st.info("左のサイドバーで条件を設定し、キーワードを入力して『検索を実行』してください。")
