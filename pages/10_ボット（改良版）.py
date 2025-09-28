# pages/10_ボット（改良版）.py
# ============================================
# この版は「gpt-5 / gpt-4.1」専用（Responses API のみ）
# 埋め込みバックエンドは OpenAI のみ（local を廃止）
# ============================================

# ============================================
# この版は「gpt-5 / gpt-4.1」専用（Responses API のみ）
# - Chat Completions 分岐と temperature UI を削除して簡素化
# - max_output_tokens / usage.input_tokens, output_tokens のみ使用
# ============================================

# ============================================
# 変更点（この版での修正・追加）
# --------------------------------------------
# 1) モデル選択の統一（pricing の一覧を使用、既定 gpt-5-mini）
# 2) Responses系（gpt-5*, gpt-4.1*）に自動対応（role分離、max_output_tokens、温度無効 等）
# 3) 料金計算は lib/costs.py に集約（このページは呼ぶだけ）
# 4) 同点スコア時のヒープ比較バグ回避（タイブレーク）
# 5) OpenAIキー: secrets.toml → env を継続
# 6) 既存機能維持（[[files: ...]]、シャード選択、RAG検索、Retrieve-only）
# 7) サンプル質問UI維持
# 8) 回答品質改善: build_prompt(strict=False)
# 9) 為替入力はサイドバーから削除。常に DEFAULT_USDJPY を使用。
# ============================================

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import random
import heapq
from itertools import count
import unicodedata
import re

import streamlit as st
import numpy as np
from openai import OpenAI

from lib.rag_utils import EmbeddingStore, NumpyVectorDB
from config.path_config import PATHS
from lib.text_normalize import normalize_ja_text
from config.sample_questions import SAMPLES, ALL_SAMPLES

from lib.costs import (
    MODEL_PRICES_USD, EMBEDDING_PRICES_USD, DEFAULT_USDJPY,
    ChatUsage, estimate_chat_cost, estimate_embedding_cost, usd_to_jpy,
)

from lib.prompts.bot_prompt import build_prompt

# ========= /1M → /1K 変換（表示用） =========
MODEL_PRICES_PER_1K: Dict[str, Dict[str, float]] = {
    m: {"in": float(p.get("in", 0.0)) / 1000.0, "out": float(p.get("out", 0.0)) / 1000.0}
    for m, p in MODEL_PRICES_USD.items()
}

# ========= パス =========
VS_ROOT: Path = PATHS.vs_root  # 例: <project>/data/vectorstore

# ========= ユーティリティ =========
def _count_tokens(text: str, model_hint: str = "cl100k_base") -> int:
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model_hint)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / 4))

def _fmt_source(meta: Dict[str, Any]) -> str:
    f = str(meta.get("file", "") or "")
    p = meta.get("page", None)
    cid = str(meta.get("chunk_id", "") or "")
    base = f"{f} p.{int(p)}" if (f and p is not None) else (f or "(unknown)")
    return f"{base} ({cid})" if cid else base

def _list_shard_dirs_openai() -> List[Path]:
    base = VS_ROOT / "openai"
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])

def _norm_path(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().replace("\\", "/")
    return s.lower()

def _get_openai_api_key() -> str | None:
    try:
        ok = st.secrets.get("openai", {}).get("api_key")
        if ok:
            return str(ok)
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")

# ---------- モデル候補（Responses専用） ----------
RESPONSES_MODELS = [m for m in MODEL_PRICES_PER_1K.keys() if m.startswith("gpt-5") or m.startswith("gpt-4.1")]

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Chat Bot (Sharded)", page_icon="💬", layout="wide")
st.title("💬 Internal Bot (RAG, Shards)")

if "q" not in st.session_state:
    st.session_state.q = ""

def _set_question(text: str):
    st.session_state.q = text

with st.sidebar:
    st.header("設定")

    # 回答モデル選択（gpt-5 / gpt-4.1 限定）
    st.markdown("### 回答モデル（Responses API）")
    all_models_sorted = sorted(RESPONSES_MODELS, key=lambda x: (0 if x.startswith("gpt-5") else 1, x))
    default_idx = all_models_sorted.index("gpt-5-mini") if "gpt-5-mini" in all_models_sorted else 0
    chat_model = st.selectbox("モデルを選択", all_models_sorted, index=default_idx)

    # 検索件数
    top_k = st.slider("検索件数（Top-K）", 1, 12, 6, 1)

    # 回答スタイル
    label_to_value = {"簡潔":"concise","標準":"standard","詳細":"detailed","超詳細":"very_detailed"}
    detail_label = st.selectbox("詳しさ", list(label_to_value.keys()), index=2)
    detail = label_to_value[detail_label]

    cite = st.checkbox("出典を角括弧で引用（[S1] 等）", value=True)

    # 出力トークン上限（Responses は max_output_tokens）
    # max_tokens = st.slider("最大出力トークン（目安）", 256, 4000, 1200, 64)
    max_tokens = st.slider("最大出力トークン（目安）", 1000, 40000, 12000, 500)

    answer_backend = st.radio("回答生成", ["OpenAI", "Retrieve-only"], index=0)
    sys_inst = st.text_area("System Instruction", "あなたは優秀な社内のアシスタントです.", height=80)

    # シャード選択（OpenAI 埋め込み専用）
    st.divider()
    st.subheader("検索対象シャード（OpenAI）")
    shard_dirs_all = _list_shard_dirs_openai()
    shard_ids_all = [p.name for p in shard_dirs_all]
    target_shards = st.multiselect("（未選択=すべて）", shard_ids_all, default=shard_ids_all)

    # 参照ファイル
    st.caption("特定ファイルだけで検索したい場合: 年/ファイル名 をカンマ区切り（例: 2025/foo.pdf, 2024/bar.pdf）")
    file_whitelist_str = st.text_input("参照ファイル（任意）", value="")
    file_whitelist = {s.strip() for s in file_whitelist_str.split(",") if s.strip()}

    # OpenAIキー確認（必須）
    has_key = bool(_get_openai_api_key())
    if not has_key:
        st.error("OpenAI APIキーが secrets.toml / 環境変数にありません。埋め込みと回答生成の双方に必須です。")

    # 🧪 サンプル質問
    st.divider()
    st.subheader("🧪 デモ用サンプル質問")
    cat = st.selectbox("カテゴリを選択", ["（未選択）"] + list(SAMPLES.keys()))
    sample = ""
    if cat != "（未選択）":
        sample = st.selectbox("サンプル質問を選択", ["（未選択）"] + SAMPLES[cat])
    else:
        st.caption("カテゴリを選ぶか、下のランダム挿入を使えます。")

    cols_demo = st.columns(2)
    with cols_demo[0]:
        st.button("⬇️ この質問を入力欄へセット", use_container_width=True,
                  disabled=(sample in ("", "（未選択）")), on_click=lambda: _set_question(sample))
    with cols_demo[1]:
        st.button("🎲 ランダム挿入", use_container_width=True,
                  on_click=lambda: _set_question(random.choice(ALL_SAMPLES)))
    send_now = st.button("🚀 サンプルで即送信", use_container_width=True,
                         disabled=(st.session_state.q.strip() == ""))

# 入力欄
q = st.text_area("質問を入力", value=st.session_state.q,
                 placeholder="この社内ボットに質問してください…", height=100)
if q != st.session_state.q:
    st.session_state.q = q

go_click = st.button("送信", type="primary")
go = go_click or bool(locals().get("send_now"))

# =========================
# 実行
# =========================
if go and st.session_state.q.strip():
    # OpenAI キー必須（埋め込みクエリ生成に必要）
    api_key = _get_openai_api_key()
    if not api_key:
        st.stop()

    try:
        # --- 検索 ---
        with st.spinner("検索中…"):
            if "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = api_key

            # ベクトルストアのルート（OpenAI 固定）
            vs_backend_dir = VS_ROOT / "openai"
            if not vs_backend_dir.exists():
                st.warning(f"ベクトルが見つかりません（{vs_backend_dir}）。先に **ベクトル化** を OpenAI で実行してください。")
                st.stop()

            shard_dirs_all = _list_shard_dirs_openai()
            selected = [vs_backend_dir / s for s in target_shards] if target_shards else [vs_backend_dir / p.name for p in shard_dirs_all]
            shard_dirs = [p for p in selected if p.is_dir() and (p / "vectors.npy").exists()]
            if not shard_dirs:
                st.warning("検索可能なシャードがありません。対象シャードのベクトル化を先に実行してください。")
                st.stop()

            # インライン参照ファイル: [[files: ...]]
            inline = re.search(r"\[\[\s*files\s*:\s*([^\]]+)\]\]", st.session_state.q, flags=re.IGNORECASE)
            inline_files = set()
            if inline:
                inline_files = {s.strip() for s in inline.group(1).split(",") if s.strip()}

            effective_whitelist = {_norm_path(x) for x in (set(file_whitelist) | set(inline_files))}
            clean_q = re.sub(r"\[\[\s*files\s*:[^\]]+\]\]", "", st.session_state.q, flags=re.IGNORECASE).strip()

            # クエリ正規化 & 埋め込み
            question = normalize_ja_text(clean_q)
            estore = EmbeddingStore(backend="openai")
            emb_tokens = _count_tokens(question, model_hint="text-embedding-3-large")
            qv = estore.embed([question]).astype("float32")  # shape=(1, d)

            # 各シャード top-k → 全体マージ
            K = int(top_k)
            heap_: List[Tuple[float, int, int, Dict[str, Any]]] = []
            tiebreak = count()

            for shp in shard_dirs:
                try:
                    vdb = NumpyVectorDB(shp)
                    hits = vdb.search(qv, top_k=K, return_="similarity")
                    for h in hits:
                        if isinstance(h, tuple) and len(h) == 3:
                            row_idx, score, meta = h
                        elif isinstance(h, tuple) and len(h) == 2:
                            score, meta = h
                            row_idx = -1
                        else:
                            continue

                        md = dict(meta or {})
                        md["shard_id"] = shp.name

                        if effective_whitelist:
                            if _norm_path(str(md.get("file", ""))) not in effective_whitelist:
                                continue

                        sc = float(score)
                        tb = next(tiebreak)

                        if len(heap_) < K:
                            heapq.heappush(heap_, (sc, tb, row_idx, md))
                        else:
                            if sc > heap_[0][0]:
                                heapq.heapreplace(heap_, (sc, tb, row_idx, md))
                except Exception as e:
                    st.warning(f"シャード {shp.name} の検索でエラー: {e}")

            raw_hits = [(rid, sc, md) for (sc, _tb, rid, md) in sorted(heap_, key=lambda x: x[0], reverse=True)]

        if not raw_hits:
            if effective_whitelist:
                st.warning("指定された参照ファイル内で該当コンテキストが見つかりませんでした。年/ファイル名（例: 2025/foo.pdf）をご確認ください。")
            else:
                st.warning("該当コンテキストが見つかりませんでした。チャンクサイズや Top-K を調整して再試行してください。")
            st.stop()

        # --- 回答生成（Responses API） ---
        chat_prompt_tokens = 0
        chat_completion_tokens = 0
        answer = None

        use_answer_backend = "Retrieve-only" if (answer_backend == "OpenAI" and not api_key) else answer_backend

        if use_answer_backend == "OpenAI":
            with st.spinner("回答生成中…"):
                labeled_contexts = [
                    f"[S{i}] {meta.get('text','')}\n[meta: {_fmt_source(meta)} / score={float(score):.3f}]"
                    for i, (_rid, score, meta) in enumerate(raw_hits, 1)
                ]
                prompt = build_prompt(
                    question,
                    labeled_contexts,
                    sys_inst=sys_inst,
                    style_hint=detail,
                    cite=cite,
                    strict=False,
                )

                client = OpenAI(api_key=api_key or "")
                try:
                    resp = client.responses.create(
                        model=chat_model,
                        input=[
                            {"role": "system", "content": sys_inst},
                            {"role": "user", "content": prompt},
                        ],
                        max_output_tokens=int(max_tokens),
                    )
                except TypeError:
                    resp = client.responses.create(
                        model=chat_model,
                        input=[
                            {"role": "system", "content": sys_inst},
                            {"role": "user", "content": prompt},
                        ],
                    )

                try:
                    answer = resp.output_text
                except Exception:
                    try:
                        answer = resp.output[0].content[0].text
                    except Exception:
                        answer = str(resp)

                try:
                    usage = getattr(resp, "usage", None)
                    if usage:
                        chat_prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                        chat_completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                except Exception:
                    pass

            st.subheader("🧠 回答")
            st.write(answer)
        else:
            st.subheader("🧩 取得のみ（要約なし）")
            st.info("Retrieve-only モードです。下の参照コンテキストを参照してください。")

        # --- 料金計算（lib/costs.py を使用） ---
        with st.container():
            emb_cost_usd = estimate_embedding_cost("text-embedding-3-large", emb_tokens)["usd"]
            chat_cost_usd = 0.0
            if use_answer_backend == "OpenAI":
                chat_cost_usd = estimate_chat_cost(
                    chat_model,
                    ChatUsage(input_tokens=chat_prompt_tokens, output_tokens=chat_completion_tokens)
                )["usd"]

            total_usd = emb_cost_usd + chat_cost_usd
            total_jpy = usd_to_jpy(total_usd, rate=DEFAULT_USDJPY)

            st.markdown("### 💴 使用料の概算（lib/costs による集計）")
            cols = st.columns(3)
            with cols[0]:
                st.metric("合計 (JPY)", f"{total_jpy:,.2f} 円")
                st.caption(f"為替 {DEFAULT_USDJPY:.2f} JPY/USD")
            with cols[1]:
                st.write("**内訳 (USD)**")
                st.write(f"- Embedding: `${emb_cost_usd:.6f}`  ({emb_tokens} tok)")
                if use_answer_backend == "OpenAI":
                    st.write(f"- Chat 合計: `${chat_cost_usd:.6f}` "
                             f"(in={chat_prompt_tokens} tok / out={chat_completion_tokens} tok)")
                st.write(f"- 合計: `${total_usd:.6f}`")
            with cols[2]:
                emb_price_per_1k = float(EMBEDDING_PRICES_USD.get("text-embedding-3-large", 0.0)) / 1000.0
                st.write("**単価 (USD / 1K tok)**")
                st.write(f"- Embedding: `${emb_price_per_1k:.5f}`（text-embedding-3-large）")
                st.write(f"- Chat 入力: `${MODEL_PRICES_PER_1K.get(chat_model,{}).get('in',0.0):.5f}`（{chat_model}）")
                st.write(f"- Chat 出力: `${MODEL_PRICES_PER_1K.get(chat_model,{}).get('out',0.0):.5f}`（{chat_model}）")

        # --- 参照コンテキスト ---
        with st.expander("🔍 参照コンテキスト（上位ヒット）", expanded=False):
            for i, (_rid, score, meta) in enumerate(raw_hits, 1):
                txt = str(meta.get("text", "") or "")
                src_label = _fmt_source(meta)
                snippet = (txt[:1000] + "…") if len(txt) > 1000 else txt
                st.markdown(f"**[S{i}] score={float(score):.3f}**  `{src_label}`\n\n{snippet}")

    except Exception as e:
        st.error(f"検索/生成中にエラー: {e}")
else:
    st.info("質問を入力して『送信』を押してください。サイドバーでシャードや回答設定、参照ファイルを調整できます。")
