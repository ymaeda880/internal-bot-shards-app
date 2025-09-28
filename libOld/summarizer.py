# lib/summarizer.py
from __future__ import annotations
from typing import List
import re
import streamlit as st
from openai import OpenAI

from lib.openai_utils import (
    is_gpt5, count_tokens, truncate_by_tokens,
    responses_generate, responses_text,
    chat_complete_safely, extract_text_from_chat,
)
from lib.text_utils import strip_html

_SUM_RES_KEY = "kw_last_summary_result"
_SUM_ERR_KEY = "kw_last_summary_error"

# ---- ローカル抽出サマリ（OpenAI不使用の保険） -------------------
def _local_summary(labelled_snips: List[str], max_sent: int = 10) -> str:
    text = "\n\n".join(labelled_snips)
    text = re.sub(r"(?m)^---\s*$", "", text)
    text = re.sub(r"(?m)^#\s*Source:.*$", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    parts = re.split(r"[。．.!?！？]\s*|\n+", text)
    uniq, out = set(), []
    for p in parts:
        p = p.strip()
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

# ---- 予算に収まるようにスニペットを削る/切り詰める（1回投げ保証） ---
def _fit_to_budget(snips: List[str], *, model: str, sys_prompt: str, user_prefix: str,
                   want_output: int, context_limit: int, safety_margin: int) -> List[str]:
    # できるだけ保持しつつ、超過したら末尾から削る
    while True:
        toks = count_tokens(sys_prompt, model) + count_tokens(user_prefix, model)
        toks += sum(count_tokens(s, model) for s in snips)
        need = toks + want_output + safety_margin
        if need <= context_limit or not snips:
            break
        snips = snips[:-1]
    # 1要素が巨大なら切り詰め
    if snips:
        budget = context_limit - (count_tokens(sys_prompt, model) + count_tokens(user_prefix, model) + want_output + safety_margin)
        budget = max(500, budget)
        snips = [s if count_tokens(s, model) <= budget else truncate_by_tokens(s, budget, model) for s in snips]
    return snips

# ---- メイン：その場で確実に描画する超シンプル実装 -------------------
def run_summary(
    *, df, model: str, temperature: float, max_tokens: int,
    topn_snippets: int, auto_batch: bool, verbose_log: bool, debug_mode: bool,
    sys_prompt: str, user_prompt_tpl: str, OPENAI_API_KEY: str, query: str
):
    st.divider()
    st.subheader("🧠 生成要約（ワンショット・確実表示）")

    # 直近結果の表示とクリアボタン（任意）
    if _SUM_RES_KEY in st.session_state:
        with st.expander("直近の要約結果（開閉可）", expanded=False):
            st.markdown(st.session_state[_SUM_RES_KEY])
        cols = st.columns([1,4])
        with cols[0]:
            if st.button("🧹 要約結果をクリア", key="clear_last_summary"):
                st.session_state.pop(_SUM_RES_KEY, None)
                st.session_state.pop(_SUM_ERR_KEY, None)

    # ---- スニペット整形（シンプル）----
    take_n = int(topn_snippets)
    selected = df.head(take_n).copy()
    labelled_snips: List[str] = []
    for _, r in selected.iterrows():
        src = f"{r.get('file')} p.{r.get('page')} (score={r.get('score')})"
        snip = strip_html(str(r.get("text","")))
        labelled_snips.append(f"---\n# Source: {src}\n{snip}")

    # 実行ボタン（フォームやチェックは使わない：再実行の副作用を排除）
    run = st.button("🧠 要約を実行（即時）", type="primary", use_container_width=True, key="run_summary_now")

    # 押されなければ何もしない
    if not run:
        st.info("『🧠 要約を実行（即時）』を押すと、このページの実行内で要約を生成して下に表示します。")
        return

    # ---- ここから“同じ実行内”で生成・描画 -------------------------
    model_hint = model
    context_limit, safety_margin = (128_000, 2_000) if is_gpt5(model) else (128_000, 1_000)
    user_prefix = user_prompt_tpl.replace("{snippets}", "").format(query=query, snippets="")

    fitted_snips = _fit_to_budget(
        labelled_snips, model=model_hint, sys_prompt=sys_prompt, user_prefix=user_prefix,
        want_output=int(max_tokens), context_limit=context_limit, safety_margin=safety_margin
    )
    if not fitted_snips:
        # それでも無理ならローカル抽出
        out_text = _local_summary(labelled_snips, max_sent=12)
        st.session_state[_SUM_RES_KEY] = out_text
        st.markdown(out_text)
        return

    snippets_text = "\n\n".join(fitted_snips)
    user_prompt = user_prompt_tpl.format(query=query, snippets=snippets_text)

    approx = count_tokens(user_prompt, model_hint) + count_tokens(sys_prompt, model_hint)
    st.caption(f"（推定入力 ~{approx:,} tok / 出力上限 {int(max_tokens):,} tok / コンテキスト~{context_limit:,} tok）")

    # ---- OpenAI 実行（失敗時は必ずローカル抽出を表示） -------------
    try:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY が未設定です。")

        client = OpenAI(api_key=OPENAI_API_KEY)

        with st.spinner("要約を生成中…"):
            if is_gpt5(model):
                resp = responses_generate(
                    client, model=model, temperature=float(temperature),
                    max_output_tokens=int(max_tokens), system_prompt=sys_prompt,
                    user_prompt=user_prompt
                )
                out = responses_text(resp)
            else:
                resp = chat_complete_safely(
                    client, model=model, temperature=float(temperature),
                    limit_tokens=int(max_tokens), system_prompt=sys_prompt,
                    user_prompt=user_prompt
                )
                out = extract_text_from_chat(resp)

        if not out or not str(out).strip():
            raise RuntimeError("モデル出力が空でした。")

        out_text = str(out).strip()
        st.session_state[_SUM_RES_KEY] = out_text  # 次回以降も残す
        st.markdown(out_text)

    except Exception as e:
        if debug_mode:
            st.error(f"OpenAI エラー: {type(e).__name__}: {e}", icon="🛑")
        # 保険：必ずローカル抽出を出す
        out_text = _local_summary(fitted_snips, max_sent=12)
        st.session_state[_SUM_RES_KEY] = out_text
        st.session_state[_SUM_ERR_KEY] = f"{type(e).__name__}: {e}"
        st.markdown(out_text)
