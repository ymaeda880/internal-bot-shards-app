# pages/20_キーワード検索.py
# ------------------------------------------------------------
# 🔎 meta.jsonl 横断検索 + （任意）OpenAI 生成要約（ボタンで実行）
# - バックエンドは OpenAI 固定（vectorstore/openai）
# - 検索結果（表・スニペット）は常に表示されたまま
# - 要約は「🧠 要約を生成」ボタン押下時のみ実行（OpenAI未設定や失敗時はローカル抽出に自動フォールバック）
# ------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Iterable, Any
import re, json, os, traceback
import pandas as pd
import streamlit as st
from lib.text_normalize import normalize_ja_text
from config.path_config import PATHS  # ← 追加：PATHSに一本化

# OpenAI は任意（生成要約で使用）
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# ============== パス（PATHSに統一） ==============
BASE_DIR: Path = PATHS.vs_root / "openai"   # ★ OpenAI 固定

# ============== 基本UI ==============
st.set_page_config(page_title="20 キーワード検索（meta横断 / OpenAI）", page_icon="🔎", layout="wide")
st.title("🔎 キーワード検索（meta.jsonl 横断 / OpenAI）")

# ============== ユーティリティ ==============
def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")

def make_snippet(text: str, pats: List[re.Pattern], total_len: int = 240) -> str:
    pos = next((m.span() for p in pats if (m := p.search(text))), (0, min(len(text), total_len)))
    left, right = max(0, pos[0] - total_len // 2), min(len(text), pos[1] + total_len // 2)
    snip = text[left:right]
    for p in pats:
        try:
            snip = p.sub(lambda m: f"<mark>{m.group(0)}</mark>", snip)
        except re.error:
            pass
    return ("…" if left else "") + snip + ("…" if right < len(text) else "")

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
    if not enc: return max(1, len(text or "") // 4)
    try: return len(enc.encode(text or ""))
    except Exception: return max(1, len(text or "") // 4)

def truncate_by_tokens(text: str, max_tokens: int, model_hint: str = "gpt-5-mini") -> str:
    enc = _encoding_for(model_hint)
    if not enc: return (text or "")[: max(100, max_tokens * 4)]
    try:
        toks = enc.encode(text or "")
        return text if len(toks) <= max_tokens else enc.decode(toks[:max_tokens])
    except Exception:
        return (text or "")[: max(100, max_tokens * 4)]

def is_gpt5(model_name: str) -> bool:
    return (model_name or "").lower().startswith("gpt-5")

def iter_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists(): return
    with path.open("r", encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if s:
                try: yield json.loads(s)
                except Exception: pass

def compile_terms(q: str, *, use_regex: bool, case_sensitive: bool, normalize_query: bool) -> List[re.Pattern]:
    if normalize_query: q = normalize_ja_text(q)
    terms = [t for t in q.split() if t]
    flags = 0 if case_sensitive else re.IGNORECASE
    pats: List[re.Pattern] = []
    for t in terms:
        try:
            pats.append(re.compile(t if use_regex else re.escape(t), flags))
        except re.error:
            pats.append(re.compile(re.escape(t), flags))
    return pats

def local_summary(labelled_snips: List[str], max_sent: int = 10) -> str:
    text = re.sub(r"<[^>]+>", "", "\n\n".join(labelled_snips))
    text = re.sub(r"(?m)^---\s*$|(?m)^#\s*Source:.*$", "", text)
    parts = [p.strip() for p in re.split(r"[。．.!?！？]\s*|\n+", text) if len((p or "").strip()) >= 6]
    out, seen = [], set()
    for p in parts:
        if p in seen: continue
        seen.add(p); out.append(f"・{p}")
        if len(out) >= max_sent: break
    if not out: return "（ローカル抽出サマリ：要約できる文が見つかりませんでした）"
    short = (parts[0][:120] + "…") if parts else ""
    return "### （ローカル抽出サマリ）\n" + "\n".join(out) + f"\n\n— 短いまとめ: {short}"

def fit_to_budget(snips: List[str], *, model: str, sys_prompt: str, user_prefix: str,
                  want_output: int, context_limit: int, safety_margin: int) -> List[str]:
    while True:
        toks = count_tokens(sys_prompt, model) + count_tokens(user_prefix, model) + sum(count_tokens(s, model) for s in snips)
        if toks + want_output + safety_margin <= context_limit or not snips: break
        snips.pop()  # 末尾から間引く
    if snips:
        budget = max(500, context_limit - (count_tokens(sys_prompt, model) + count_tokens(user_prefix, model) + want_output + safety_margin))
        snips = [s if count_tokens(s, model) <= budget else truncate_by_tokens(s, budget, model) for s in snips]
    return snips

def openai_summary(*, model: str, temperature: float, max_tokens: int,
                   sys_prompt: str, user_prompt: str, api_key: str) -> str:
    if not (_HAS_OPENAI and api_key): raise RuntimeError("OpenAI未設定")
    client = OpenAI(api_key=api_key)
    if is_gpt5(model):
        resp = client.responses.create(
            model=model,
            input=[{"role": "system", "content": sys_prompt.strip()}, {"role": "user", "content": user_prompt}],
            temperature=float(temperature), max_output_tokens=int(max_tokens),
        )
        return getattr(resp, "output_text", "") or ""
    else:
        resp = client.chat.completions.create(
            model=model, temperature=float(temperature), max_tokens=int(max_tokens),
            messages=[{"role": "system", "content": sys_prompt.strip()}, {"role": "user", "content": user_prompt}],
        )
        return (resp.choices[0].message.content or "") if getattr(resp, "choices", None) else ""

# ============== サイドバー（OpenAI 固定） ==============
with st.sidebar:
    st.header("検索対象（OpenAI）")
    if not BASE_DIR.exists():
        st.error(f"{BASE_DIR} が見つかりません。先に 03 ベクトル化を OpenAI で実行してください。"); st.stop()

    shard_ids = [p.name for p in sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])]
    sel_shards = st.multiselect("対象シャード", shard_ids, default=shard_ids)

    st.divider(); st.subheader("絞り込み（任意）")
    year_min = st.number_input("年（下限）", value=0, step=1, help="0 で無効")
    year_max = st.number_input("年（上限）", value=9999, step=1, help="9999 で無効")
    file_filter = st.text_input("ファイル名フィルタ（部分一致）", value="").strip()

    st.divider(); st.subheader("表示設定")
    max_rows = st.number_input("最大表示件数", min_value=50, max_value=5000, value=500, step=50)
    snippet_len = st.slider("スニペット長（前後合計）", min_value=80, max_value=800, value=240, step=20)
    show_cols = st.multiselect(
        "表示カラム", ["file","year","page","shard_id","chunk_id","chunk_index","score","text"],
        default=["file","year","page","shard_id","score","text"]
    )

    st.divider(); st.subheader("🧠 生成オプション")
    def get_openai_key() -> str | None:
        try:
            return st.secrets.get("OPENAI_API_KEY") or (st.secrets.get("openai") or {}).get("api_key") or os.getenv("OPENAI_API_KEY")
        except Exception:
            return os.getenv("OPENAI_API_KEY")
    OPENAI_API_KEY = get_openai_key()

    model = st.selectbox("モデル", ["gpt-5-mini", "gpt-5", "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temperature = 1.0 if is_gpt5(model) else st.slider("temperature", 0.0, 1.0, 0.2, 0.05)

    # 出力トークン上限（Responses は max_output_tokens）
    # max_tokens = st.slider("出力トークン上限", 128, 32000, 2000, 128)
    max_tokens = st.slider("最大出力トークン（目安）", 1000, 40000, 12000, 500)

    topn_snippets = st.slider("要約に使う上位スニペット数", 5, 200, 30, 5)
    sys_prompt = st.text_area("System Prompt",
        "あなたは事実に忠実なリサーチアシスタントです。根拠のある記述のみを日本語で簡潔にまとめてください。", height=80)
    user_prompt_tpl = st.text_area("User Prompt テンプレ（{query}, {snippets} を埋め込み）",
        "以下はキーワード検索で得られたヒットスニペットです。この情報【のみ】を根拠に、"
        "クエリ『{query}』について要点を箇条書き→短いまとめの順で整理してください。\n\n# ヒットスニペット\n{snippets}",
        height=120)
    debug_mode = st.checkbox("デバッグ情報を表示", value=False)

# ============== 検索フォーム ==============
st.markdown("### クエリ")
c1, c2 = st.columns([3,2])
with c1: query = st.text_input("キーワード（空白区切りで AND / OR 指定）", value="")
with c2: bool_mode = st.radio("モード", ["AND", "OR"], index=0, horizontal=True)

c3, c4, c5, c6 = st.columns(4)
with c3: use_regex = st.checkbox("正規表現", value=False)
with c4: case_sensitive = st.checkbox("大文字小文字を区別", value=False)
with c5: normalize_query = st.checkbox("日本語スペース正規化（推奨）", value=True)
with c6: norm_body = st.checkbox("本文も正規化して検索", value=True, help="取り込み時に正規化していないコーパス向け")

go = st.button("検索を実行", type="primary")

# ============== 検索の実行（結果は session_state に保存） ==============
if go:
    try:
        if not sel_shards: st.warning("少なくとも1つのシャードを選択してください。"); st.stop()
        pats = compile_terms(query, use_regex=use_regex, case_sensitive=case_sensitive, normalize_query=normalize_query)
        if not pats: st.warning("検索語が空です。キーワードを入力してください。"); st.stop()

        rows: List[Dict[str,Any]] = []
        total_scanned = 0

        for sid in sel_shards:
            meta_path = BASE_DIR / sid / "meta.jsonl"
            if not meta_path.exists():
                st.warning(f"{meta_path} が見つかりません。スキップします。"); continue
            for obj in iter_jsonl(meta_path):
                total_scanned += 1
                yr = obj.get("year")
                if isinstance(yr, int):
                    if year_min and yr < year_min: continue
                    if year_max < 9999 and yr > year_max: continue
                if (f := str(obj.get("file",""))) and file_filter and file_filter.lower() not in f.lower():
                    continue

                text = str(obj.get("text",""))
                tgt = normalize_ja_text(text) if norm_body else text
                ok = all(p.search(tgt) for p in pats) if bool_mode == "AND" else any(p.search(tgt) for p in pats)
                if not ok: continue

                score = sum(1 for p in pats for _ in p.finditer(tgt))
                rows.append({
                    "file": obj.get("file"), "year": obj.get("year"), "page": obj.get("page"),
                    "shard_id": obj.get("shard_id", sid), "chunk_id": obj.get("chunk_id"),
                    "chunk_index": obj.get("chunk_index"), "score": int(score),
                    "text": make_snippet(text, pats, total_len=int(snippet_len)),
                })
                if len(rows) >= int(max_rows): break
            if len(rows) >= int(max_rows): break

        if not rows: st.warning("ヒットなし。検索語やフィルタを調整してください。"); st.stop()

        # 検索結果と設定を保存（要約ボタン後の再描画で使う）
        st.session_state["kw_rows"] = rows
        st.session_state["kw_scanned"] = total_scanned
        st.session_state["kw_show_order"] = [c for c in show_cols if c in rows[0].keys()] or ["file","year","page","shard_id","score","text"]
        st.session_state["kw_sort_cols"] = ["score","year","file","page"]
        st.session_state["kw_query"] = query

        # 要約用の設定も保存
        st.session_state["kw_gen_cfg"] = dict(
            OPENAI_API_KEY=OPENAI_API_KEY, model=model, temperature=float(temperature),
            max_tokens=int(max_tokens), topn=int(topn_snippets),
            sys_prompt=sys_prompt, user_prompt_tpl=user_prompt_tpl,
        )

        st.success(f"ヒット {len(rows):,d} 件 / 走査 {total_scanned:,d} レコード（上位のみ表示）")

    except Exception:
        st.error("検索処理でエラーが発生しました。", icon="🛑")
        if st.session_state.get("kw_gen_cfg", {}).get("debug_mode", False) or debug_mode:
            st.code("".join(traceback.format_exc()))

# ============== 共通描画ブロック（検索直後 / ボタン押下後の両方で実行） ==============
if st.session_state.get("kw_rows"):
    rows_saved: List[Dict[str, Any]] = st.session_state["kw_rows"]
    sort_cols = st.session_state.get("kw_sort_cols", ["score","year","file","page"])
    show_order = st.session_state.get("kw_show_order", ["file","year","page","shard_id","score","text"])
    df = pd.DataFrame(rows_saved).sort_values(sort_cols, ascending=[False, True, True, True])

    # 1) ヒット一覧（表）
    st.dataframe(df[[c for c in show_order if c != "text"]], use_container_width=True, height=420)

    # 2) CSV ダウンロード
    csv_bytes = df[show_order].to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 CSV をダウンロード", data=csv_bytes, file_name="keyword_hits.csv", mime="text/csv")

    # 3) ヒットスニペット（常に表示されたまま）
    if "text" in show_order:
        st.divider()
        with st.expander("ヒットスニペット（クリックで展開）", expanded=False):
            for i, row in df.head(200).iterrows():
                colA, colB = st.columns([4,1])
                with colA:
                    st.markdown(
                        f"**{row.get('file')}**  year={row.get('year')}  p.{row.get('page')}  score={row.get('score')}",
                        help=row.get("chunk_id")
                    )
                    st.markdown(row.get("text",""), unsafe_allow_html=True)
                with colB:
                    payload = json.dumps(str(row.get("file")), ensure_ascii=False)
                    st.components.v1.html(f"""
                    <button id="cpy_{i}" style="padding:6px 10px;border-radius:8px;border:1px solid #dadce0;background:#fff;cursor:pointer;font-size:0.9rem;">📋 year/file をコピー</button>
                    <script>
                      const b=document.getElementById("cpy_{i}");
                      b&&b.addEventListener("click",async()=>{{
                        try{{await navigator.clipboard.writeText({payload});
                          const o=b.innerText;b.innerText="✅ コピーしました";setTimeout(()=>{{b.innerText=o}},1200);
                        }}catch(e){{alert("コピーに失敗: "+e)}}
                      }});
                    </script>
                    """, height=38)

    # 4) 🧠 要約ボタン
    st.divider()
    gen_clicked = st.button("🧠 要約を生成", type="primary", use_container_width=True)

    if gen_clicked:
        # 保存済み設定の読込
        cfg = st.session_state.get("kw_gen_cfg", {})
        OPENAI_API_KEY = cfg.get("OPENAI_API_KEY")
        model = cfg.get("model", "gpt-5-mini")
        temperature = cfg.get("temperature", 1.0 if is_gpt5(model) else 0.2)
        max_tokens = cfg.get("max_tokens", 2000)
        topn_snippets = cfg.get("topn", 30)
        sys_prompt = cfg.get("sys_prompt", "あなたは事実に忠実なリサーチアシスタントです。日本語で簡潔にまとめてください。")
        user_prompt_tpl = cfg.get("user_prompt_tpl", "クエリ『{query}』\n{snippets}")
        query = st.session_state.get("kw_query", "")

        # スニペット準備
        labelled = [
            f"---\n# Source: {r.get('file')} p.{r.get('page')} (score={r.get('score')})\n{strip_html(str(r.get('text','')))}"
            for _, r in df.head(int(topn_snippets)).iterrows()
        ]

        # 予算合わせ
        user_prefix = user_prompt_tpl.replace("{snippets}", "").format(query=query, snippets="")
        context_limit, safety_margin = (128_000, 2_000) if is_gpt5(model) else (128_000, 1_000)
        fitted = fit_to_budget(
            labelled, model=model, sys_prompt=sys_prompt, user_prefix=user_prefix,
            want_output=int(max_tokens), context_limit=context_limit, safety_margin=safety_margin
        )

        # 要約の実行（OpenAI → フォールバック）
        st.subheader("🧠 生成要約")
        if not fitted:
            with st.spinner("🧩 ローカル抽出サマリを生成中…"):
                st.markdown(local_summary(labelled, max_sent=12))
        else:
            snippets_text = "\n\n".join(fitted)
            user_prompt = user_prompt_tpl.format(query=query, snippets=snippets_text)
            approx_in = count_tokens(user_prompt, model) + count_tokens(sys_prompt, model)
            st.caption(f"（推定入力 ~{approx_in:,} tok / 出力上限 {int(max_tokens):,} tok / コンテキスト~{context_limit:,} tok）")

            try:
                with st.spinner("🧠 要約を生成中…"):
                    out = openai_summary(
                        model=model, temperature=float(temperature), max_tokens=int(max_tokens),
                        sys_prompt=sys_prompt, user_prompt=user_prompt, api_key=OPENAI_API_KEY
                    )
                if str(out).strip():
                    st.markdown(str(out).strip())
                else:
                    with st.spinner("🧩 ローカル抽出サマリを生成中…"):
                        st.info("⚠️ モデル出力が空だったため、ローカル抽出サマリを表示します。")
                        st.markdown(local_summary(fitted, max_sent=12))
            except Exception as e:
                if debug_mode:
                    st.error(f"OpenAI エラー: {type(e).__name__}: {e}", icon="🛑")
                    st.code("".join(traceback.format_exc()))
                with st.spinner("🧩 ローカル抽出サマリを生成中…"):
                    st.markdown(local_summary(fitted, max_sent=12))

# ============== 初期ガイダンス ==============
if not st.session_state.get("kw_rows"):
    st.info("左のサイドバーで条件を設定し、キーワードを入力して『検索を実行』してください。")
