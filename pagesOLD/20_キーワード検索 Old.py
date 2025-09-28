# pages/20_キーワード検索.py
# ------------------------------------------------------------
# 🔎 meta.jsonl 横断検索 + （任意）OpenAI 生成要約
# - gpt-5 系（特に gpt-5-mini）向けに事前トークン見積 & 自動バッチ要約を実装
# - 生成前に「要約を実行」チェック + ボタンで明示確認（常に）
# - オーバー時は警告表示、処理内容を逐一UIに明示
# - gpt-5 系は Responses API（max_output_tokens）、他は Chat Completions
# - temperature: gpt-5 系は 1.0 固定（UIは metric 表示）
# - スニペットは既定で畳み表示（st.expander）
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Iterable, Any, Tuple
from datetime import datetime
import re
import json
import unicodedata
import os

import numpy as np  # noqa
import pandas as pd
import streamlit as st
from openai import OpenAI

# ============== パス ==============
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
VS_ROOT  = DATA_DIR / "vectorstore"

# ============== 日本語正規化 ==============
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
    return re.sub(r"<[^>]+>", "", s or "")

# ---- token helpers ----
def _encoding_for(model_hint: str):
    import tiktoken
    try:
        return tiktoken.encoding_for_model(model_hint)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str, model_hint: str = "gpt-5-mini") -> int:
    try:
        enc = _encoding_for(model_hint)
        return len(enc.encode(text or ""))
    except Exception:
        return max(1, int(len(text or "") / 4))

def _truncate_by_tokens(text: str, max_tokens: int, model_hint: str = "gpt-5-mini") -> str:
    try:
        enc = _encoding_for(model_hint)
        toks = enc.encode(text or "")
        if len(toks) <= max_tokens:
            return text or ""
        return enc.decode(toks[:max_tokens])
    except Exception:
        max_chars = max(100, max_tokens * 4)
        return (text or "")[:max_chars]

def _is_gpt5(model_name: str) -> bool:
    return (model_name or "").lower().startswith("gpt-5")

# ---- JSONL ----
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

# ============== UI 基本 ==============
st.set_page_config(page_title="20 キーワード検索（meta横断）", page_icon="🔎", layout="wide")
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

    # ============== 生成オプション ==============
    st.divider()
    st.subheader("🧠 生成オプション（OpenAI）")

    # secrets.toml 優先、なければ環境変数をフォールバック
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
    has_key = bool(OPENAI_API_KEY)

    gen_enabled = st.checkbox("ヒット要約を生成する", value=True if has_key else False, disabled=not has_key)
    if not has_key:
        st.warning("OPENAI_API_KEY が未設定のため、生成は無効です。", icon="⚠️")

    model = st.selectbox(
        "モデル",
        ["gpt-5-mini", "gpt-5", "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
        disabled=not gen_enabled
    )

    # gpt-5 系は temperature=1 固定（Sliderはエラーになるので metric 表示）
    if _is_gpt5(model):
        temperature = 1.0
        st.metric(label="temperature", value="1.0")
        st.caption("🔒 gpt-5 系モデルでは temperature=1 に固定されます。")
    else:
        temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05, disabled=not gen_enabled)

    # 上限は広めに（モデルが対応しない値は API が弾く）
    max_tokens = st.slider("出力トークン上限", 128, 32000, 2000, 128, disabled=not gen_enabled)
    topn_snippets = st.slider("生成に使う上位スニペット数", 5, 200, 30, 5, disabled=not gen_enabled)

    auto_batch = st.checkbox("オーバー時は自動バッチ要約で処理する（推奨）", value=True, disabled=not gen_enabled)
    verbose_log = st.checkbox("バッチ処理の詳細ログを表示", value=True, disabled=not gen_enabled)

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
            alert("コピーに失敗しました: " + e);
          }}
        }});
      }}
    </script>
    """
    st.components.v1.html(html, height=38)

# ---------- OpenAI 呼び出し ----------
def _use_mct(model_name: str) -> bool:
    m = (model_name or "").lower()
    return m.startswith("gpt-5") or m.startswith("o")

def _chat_complete_safely(client: OpenAI, *, model: str, temperature: float,
                          limit_tokens: int, system_prompt: str, user_prompt: str):
    def _call(use_mct: bool):
        payload = {
            "model": model,
            "temperature": float(temperature),
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt},
            ],
        }
        if use_mct:
            payload["max_completion_tokens"] = int(limit_tokens)
        else:
            payload["max_tokens"] = int(limit_tokens)
        return client.chat.completions.create(**payload)
    prefer_mct = _use_mct(model)
    try:
        return _call(prefer_mct)
    except Exception:
        return _call(not prefer_mct)

def _extract_text_from_chat(resp_obj) -> str:
    try:
        content = resp_obj.choices[0].message.content
        return content or ""
    except Exception:
        return ""

# Responses API for gpt-5（SDK差異のため response_format は渡さない）
def _responses_generate(client: OpenAI, *, model: str, temperature: float,
                        max_output_tokens: int, system_prompt: str, user_prompt: str):
    return client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=float(temperature),
        max_output_tokens=int(max_output_tokens),
    )

def _responses_text(resp) -> str:
    try:
        txt = resp.output_text
        if isinstance(txt, str) and txt.strip():
            return txt
    except Exception:
        pass
    try:
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
    except Exception:
        pass
    return ""

# ============== バッチ分割ロジック（gpt-5-mini前提） ==============
def _gpt5mini_limits():
    # 既定：context 128k、セーフティマージン 2k
    return 128_000, 2_000

def _estimate_prompt_tokens(model: str, sys_prompt: str, user_prompt_prefix: str) -> int:
    return _count_tokens(sys_prompt, model) + _count_tokens(user_prompt_prefix, model)

def _split_into_batches(snippets: List[str], model: str, input_budget_tokens: int) -> List[List[str]]:
    batches: List[List[str]] = []
    current: List[str] = []
    cur_tokens = 0
    for s in snippets:
        t = _count_tokens(s, model)
        if t > input_budget_tokens:
            s = _truncate_by_tokens(s, input_budget_tokens, model)
            t = _count_tokens(s, model)
        if cur_tokens + t <= input_budget_tokens:
            current.append(s); cur_tokens += t
        else:
            if current:
                batches.append(current)
            current = [s]; cur_tokens = t
    if current:
        batches.append(current)
    return batches

def _render_snippets(snips: List[str]) -> str:
    return "\n\n".join(snips)

# ============== 実行 ==============
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
    non_text_cols = [c for c in show_order if c != "text"]
    st.dataframe(df[non_text_cols], use_container_width=True, height=420)

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 CSV をダウンロード", data=csv_bytes, file_name="keyword_hits.csv", mime="text/csv")

    # ============== 生成（スニペットより前に表示） ==============
    if gen_enabled:
        st.divider()
        st.subheader("🧠 生成要約（OpenAI）")

        # 1) 上位 N スニペット（HTML除去 & 1つひとつに Source ラベル）
        take_n = int(topn_snippets)
        selected = df.head(take_n).copy()

        labelled_snips: List[str] = []
        for _, r in selected.iterrows():
            src = f"{r.get('file')} p.{r.get('page')} (score={r.get('score')})"
            snip = strip_html(str(r.get("text","")))
            labelled_snips.append(f"---\n# Source: {src}\n{snip}")

        # 2) 事前見積（まずは表示だけ行う）
        model_hint = model
        context_limit, safety_margin = _gpt5mini_limits() if _is_gpt5(model) else (128_000, 1_000)
        user_prefix = user_prompt_tpl.replace("{snippets}", "").format(query=query, snippets="")
        prompt_overhead = _estimate_prompt_tokens(model_hint, sys_prompt, user_prefix)
        snippets_tokens = sum(_count_tokens(s, model_hint) for s in labelled_snips)
        want_output = int(max_tokens)
        needed_total = prompt_overhead + snippets_tokens + want_output + safety_margin

        st.caption(f"見積: prompt+snips={prompt_overhead + snippets_tokens:,} / "
                   f"出力上限={want_output:,} / safety={safety_margin:,} / "
                   f"context_limit~{context_limit:,}")

        will_overflow = needed_total > context_limit
        if will_overflow:
            over = needed_total - context_limit
            st.error(f"⚠️ 入力が大きすぎます（推定 {needed_total:,} tok が上限 {context_limit:,} tok を {over:,} tok 超過）。", icon="⚠️")
            if not auto_batch:
                st.info("対処: ① 生成に使うスニペット数を減らす ② 出力トークン上限を下げる ③ モデルを変更（大コンテキスト）")

        # 3) 実行前確認 UI（常に確認）
        with st.form("run_summary_form"):
            agree = st.checkbox("この条件で要約を実行してよい（実行前の最終確認）", value=False)
            run_summary = st.form_submit_button("🧠 要約を実行", type="primary", use_container_width=True)
        if not (agree and run_summary):
            st.info("要約を開始するには、チェックを入れて『🧠 要約を実行』を押してください。")
        else:
            # ==== ここから実際の生成 ====
            client = OpenAI(api_key=OPENAI_API_KEY)

            if will_overflow and not auto_batch:
                st.stop()

            # --- 自動バッチ要約 ---
            if will_overflow and auto_batch:
                st.info("🪄 自動バッチ要約を開始します：スニペットを複数バッチに分割 → 各バッチ要約 → 最終統合。")
                per_batch_budget = max(1000, context_limit - want_output - safety_margin - prompt_overhead)
                batches = _split_into_batches(labelled_snips, model_hint, per_batch_budget)
                st.caption(f"バッチ数: {len(batches)} / バッチ入力バジェット~{per_batch_budget:,} tok")

                batch_summaries: List[str] = []
                for bi, batch in enumerate(batches, start=1):
                    batch_snips = _render_snippets(batch)
                    user_prompt = user_prompt_tpl.format(query=query, snippets=batch_snips)
                    approx_in = _count_tokens(user_prompt, model_hint) + _count_tokens(sys_prompt, model_hint)
                    if verbose_log:
                        st.write(f"Batch {bi}/{len(batches)}: 入力推定 ~{approx_in:,} tok / 出力上限 {want_output:,} tok")
                    with st.spinner(f"Batch {bi}/{len(batches)} を要約中…"):
                        if _is_gpt5(model):
                            resp = _responses_generate(
                                client, model=model, temperature=float(temperature),
                                max_output_tokens=want_output, system_prompt=sys_prompt,
                                user_prompt=user_prompt
                            )
                            text = _responses_text(resp)
                            finish = getattr(resp, "finish_reason", None)
                            usage  = getattr(resp, "usage", None)
                        else:
                            resp = _chat_complete_safely(
                                client, model=model, temperature=float(temperature),
                                limit_tokens=want_output, system_prompt=sys_prompt,
                                user_prompt=user_prompt
                            )
                            text = _extract_text_from_chat(resp)
                            try:
                                finish = resp.choices[0].finish_reason
                                usage = resp.usage
                            except Exception:
                                finish, usage = None, None
                    if verbose_log:
                        try:
                            ct = getattr(usage, "completion_tokens", None) or usage.get("completion_tokens")
                            pt = getattr(usage, "prompt_tokens", None) or usage.get("prompt_tokens")
                            st.caption(f"Batch {bi}: finish={finish} / tokens(c/p)={ct}/{pt}")
                        except Exception:
                            st.caption(f"Batch {bi}: finish={finish}")
                    batch_summaries.append(f"[Batch {bi} 要約]\n{text.strip()}")

                # 最終統合
                st.info("🧩 バッチ要約を統合しています…")
                joined_batch = "\n\n".join(batch_summaries)
                prefix = "以下は複数バッチの要約です。重複を統合し、矛盾を解消し、最終の簡潔な日本語サマリを出力してください。\n\n"
                integration_prompt = f"{prefix}{joined_batch}"
                approx_integration = _count_tokens(integration_prompt, model_hint) + _count_tokens(sys_prompt, model_hint)
                if approx_integration + want_output + safety_margin > context_limit:
                    keep = context_limit - want_output - safety_margin - _count_tokens(sys_prompt, model_hint)
                    integration_prompt = _truncate_by_tokens(integration_prompt, max(1000, keep), model_hint)
                    approx_integration = _count_tokens(integration_prompt, model_hint) + _count_tokens(sys_prompt, model_hint)
                    st.caption(f"統合プロンプトをトリム: 入力推定 ~{approx_integration:,} tok")

                with st.spinner("最終統合要約を生成中…"):
                    if _is_gpt5(model):
                        final_resp = _responses_generate(
                            client, model=model, temperature=float(temperature),
                            max_output_tokens=want_output, system_prompt=sys_prompt,
                            user_prompt=integration_prompt
                        )
                        final_text = _responses_text(final_resp)
                        finish_final = getattr(final_resp, "finish_reason", None)
                        usage_final  = getattr(final_resp, "usage", None)
                    else:
                        final_resp = _chat_complete_safely(
                            client, model=model, temperature=float(temperature),
                            limit_tokens=want_output, system_prompt=sys_prompt,
                            user_prompt=integration_prompt
                        )
                        final_text = _extract_text_from_chat(final_resp)
                        try:
                            finish_final = final_resp.choices[0].finish_reason
                            usage_final  = final_resp.usage
                        except Exception:
                            finish_final, usage_final = None, None

                st.markdown(final_text if final_text.strip() else "_（統合結果が空でした）_")
                cols = st.columns(3)
                with cols[0]:
                    st.caption(f"finish_reason: **{finish_final}**")
                with cols[1]:
                    try:
                        ct = getattr(usage_final, "completion_tokens", None) or usage_final.get("completion_tokens")
                        pt = getattr(usage_final, "prompt_tokens", None) or usage_final.get("prompt_tokens")
                        st.caption(f"tokens (c/p): **{ct}/{pt}**")
                    except Exception:
                        pass
                with cols[2]:
                    st.caption(f"model: **{model}**")

            else:
                # --- 単発で収まる場合 ---
                snippets_text = _render_snippets(labelled_snips)
                user_prompt = user_prompt_tpl.format(query=query, snippets=snippets_text)
                approx_total = _count_tokens(user_prompt, model_hint) + _count_tokens(sys_prompt, model_hint)
                st.caption(f"（プロンプト推定トークン: ~{approx_total:,} tok / 出力上限 {want_output:,} tok）")

                with st.spinner("要約を生成中…"):
                    if _is_gpt5(model):
                        raw = _responses_generate(
                            client, model=model, temperature=float(temperature),
                            max_output_tokens=want_output, system_prompt=sys_prompt,
                            user_prompt=user_prompt
                        )
                        out_text = _responses_text(raw)
                        finish = getattr(raw, "finish_reason", None)
                        usage  = getattr(raw, "usage", None)
                    else:
                        raw = _chat_complete_safely(
                            client, model=model, temperature=float(temperature),
                            limit_tokens=want_output, system_prompt=sys_prompt,
                            user_prompt=user_prompt
                        )
                        out_text = _extract_text_from_chat(raw)
                        try:
                            finish = raw.choices[0].finish_reason
                            usage  = raw.usage
                        except Exception:
                            finish, usage = None, None

                if (not out_text or not out_text.strip()) and finish == "length":
                    st.info("🔁 出力が打ち切られたため、続きの生成を加えています…")
                    cont_prompt = user_prompt + "\n\n【続きのみを簡潔に出力してください。】"
                    if _is_gpt5(model):
                        raw2 = _responses_generate(
                            client, model=model, temperature=float(temperature),
                            max_output_tokens=want_output, system_prompt=sys_prompt,
                            user_prompt=cont_prompt
                        )
                        out_text2 = _responses_text(raw2)
                    else:
                        raw2 = _chat_complete_safely(
                            client, model=model, temperature=float(temperature),
                            limit_tokens=want_output, system_prompt=sys_prompt,
                            user_prompt=cont_prompt
                        )
                        out_text2 = _extract_text_from_chat(raw2)
                    out_text = (out_text or "") + ("\n" + out_text2 if out_text2 else "")

                st.markdown(out_text if (out_text and out_text.strip()) else "_（本文が返りませんでした）_")
                cols = st.columns(3)
                with cols[0]:
                    st.caption(f"finish_reason: **{finish}**")
                with cols[1]:
                    try:
                        ct = getattr(usage, "completion_tokens", None) or usage.get("completion_tokens")
                        pt = getattr(usage, "prompt_tokens", None) or usage.get("prompt_tokens")
                        st.caption(f"tokens (c/p): **{ct}/{pt}**")
                    except Exception:
                        pass
                with cols[2]:
                    st.caption(f"model: **{model}**")

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
                    copy_button(text=str(row.get("file")), label="year/file をコピー", key=f"cpy_{i}")

else:
    st.info("左でシャードと条件を選び、キーワードを入力して『検索を実行』してください。")
