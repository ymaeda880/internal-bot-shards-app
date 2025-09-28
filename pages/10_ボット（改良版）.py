# pages/10_ボット（新）.py

# ============================================
# 変更点（この版での修正・追加）
# --------------------------------------------
# 1) モデル選択の統一:
#    - chat/Responses 両系のモデルを pricing.py の一覧から選択可能に。
#    - 既定モデルを gpt-5-mini に設定。
# 2) Responses系（gpt-5*, gpt-4.1*）のAPI/挙動に自動対応（重要バグ修正）:
#    - ✅ role分離した input を使用（[{"role":"system"}, {"role":"user"}]）。
#    - ✅ temperature を送らない（非対応のため）。
#    - ✅ max_tokens ではなく max_output_tokens を使用。
#    - ✅ 出力抽出を堅牢化（resp.output_text → ネスト構造のフォールバック）。
#    - ✅ UI の temperature スライダーを自動で disabled に。
# 3) 料金計算の一元化:
#    - config/pricing.py の MODEL_PRICES_USD（USD/1M）/ EMBEDDING_PRICES_USD（USD/1K）/
#      DEFAULT_USDJPY を使用。
#    - 🆕 このページ内で MODEL_PRICES_USD を /1K に変換して表示・計算（MODEL_PRICES_PER_1K）。
# 4) 同点スコア時のヒープ比較バグ回避（タイブレーク）継続。
# 5) OpenAIキーの取得は secrets.toml → env を継続。
# 6) 既存機能（ファイル指定 [[files: ...]]、シャード選択、RAG検索、Retrieve-only 等）は維持。
# 7) サンプル質問UIをサイドバーに維持（カテゴリ・選択・ランダム・即送信）。
# 8) 🆕 回答品質改善: build_prompt(strict=False) にして過度な「分かりません」を抑制。
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

from lib.rag_utils import EmbeddingStore, NumpyVectorDB, build_prompt
from config.path_config import PATHS  # VS_ROOT を設定ファイルから取得
# 価格とモデル一覧を pricing.py から使用（MODEL_PRICES_USD は USD / 1M tokens）
from config.pricing import MODEL_PRICES_USD, EMBEDDING_PRICES_USD, DEFAULT_USDJPY

# ========= /1M → /1K 変換 =========
# ※ config/pricing.py は他所でも使われるため、ここでのみ /1K に換算して利用します。
MODEL_PRICES_PER_1K: Dict[str, Dict[str, float]] = {
    m: {
        "in": float(p.get("in", 0.0)) / 1000.0,
        "out": float(p.get("out", 0.0)) / 1000.0,
    }
    for m, p in MODEL_PRICES_USD.items()
}

# ========= パス =========
VS_ROOT: Path = PATHS.vs_root  # 例: <project>/data/vectorstore

# ========= 日本語テキスト正規化 =========
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

def _list_shard_dirs(backend: str) -> List[Path]:
    base = VS_ROOT / backend
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])

def _norm_path(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().replace("\\", "/")
    return s.lower()

def _get_openai_api_key() -> str | None:
    # secrets.toml -> env
    try:
        ok = st.secrets.get("openai", {}).get("api_key")
        if ok:
            return str(ok)
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")

# ---------- モデル分類 ----------
# Responses API 系（temperature 非対応、max_output_tokens を使う）
RESPONSES_MODELS = [m for m in MODEL_PRICES_PER_1K.keys() if m.startswith("gpt-5") or m.startswith("gpt-4.1")]
# Chat Completions API 系（temperature / max_tokens 使用可）
CHAT_MODELS = [m for m in MODEL_PRICES_PER_1K.keys() if m.startswith("gpt-4o") or m.startswith("gpt-3.5")]

def _use_responses_api(model_name: str) -> bool:
    return model_name in RESPONSES_MODELS

# =========================
# サンプル質問
# =========================
SAMPLES = {
    "補助金": [
        "この資料の補助対象経費・対象外経費の要件を簡潔にまとめてください。[出典を明記]",
        "この資料の補助対象者を簡潔にまとめてください。[出典を明記]",
        "申請スケジュールと主要な締切日を一覧で教えてください。",
        "中小企業者・小規模事業者の定義を比較表で示してください。",
    ],
    "環境ガイドライン": [
        "ガイドライン改定の背景を整理してください。",
        "環境報告の記載事項について教えてください。",
    ],
    "令和７年度環境省行政事業レビュー": [
        "令和７年度環境省行政事業レビューの参加者は。",
        "令和７年度環境省行政事業レビューにおいて放射線の健康管理について出た意見は？。",
        "令和７年度環境省行政事業レビューにおいて潮流発電について出た意見は？"
    ],
     "政策委員会金融政策決定会合": [
        "2023年10月30日の政策委員会金融政策決定会合の出席委員",
        "2024年1月22日の政策委員会金融政策決定会合の政府からの出席者は",
        "2024年1月23日の政策委員会金融政策決定会合の政府からの出席者を全て教えて（慎重に考えて2024年1月22 日ではなく2024年1月23日です）",
        "2023年10月30日の政策委員会金融政策決定会合で議論された為替市場動向は"
    ],
}
ALL_SAMPLES = [q for qs in SAMPLES.values() for q in qs]

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

    # 埋め込みバックエンド
    embed_backend_label = st.radio(
        "埋め込みバックエンド（作成時と一致）",
        ["local (sentence-transformers)", "openai"],
        index=1
    )
    embed_backend = "openai" if embed_backend_label.startswith("openai") else "local"

    # 回答モデル選択（pricing.py の一覧から）
    st.markdown("### 回答モデル")
    all_models_sorted = sorted(MODEL_PRICES_PER_1K.keys(), key=lambda x: (0 if x.startswith("gpt-5") else 1, x))
    default_idx = all_models_sorted.index("gpt-5-mini") if "gpt-5-mini" in all_models_sorted else 0
    chat_model = st.selectbox("モデルを選択", all_models_sorted, index=default_idx)

    # 検索件数
    top_k = st.slider("検索件数（Top-K）", 1, 12, 6, 1)

    # 回答スタイル
    label_to_value = {"簡潔":"concise","標準":"standard","詳細":"detailed","超詳細":"very_detailed"}
    detail_label = st.selectbox("詳しさ", list(label_to_value.keys()), index=2)
    detail = label_to_value[detail_label]

    cite = st.checkbox("出典を角括弧で引用（[S1] 等）", value=True)

    # 温度（Responses系は無効化）
    is_responses = _use_responses_api(chat_model)
    temperature = st.slider(
        "temperature（Chat系のみ有効）", 0.0, 1.0, 0.2, 0.05,
        disabled=is_responses, help="gpt-5*, gpt-4.1* では無効（固定）"
    )

    # 出力トークン上限
    max_tokens = st.slider("最大出力トークン（目安）", 256, 4000, 1200, 64)

    answer_backend = st.radio("回答生成", ["OpenAI", "Retrieve-only"], index=0)
    sys_inst = st.text_area("System Instruction", "あなたは優秀な社内のアシスタントです.", height=80)

    # シャード選択
    st.divider()
    st.subheader("検索対象シャード")
    shard_dirs_all = _list_shard_dirs(embed_backend)
    shard_ids_all = [p.name for p in shard_dirs_all]
    target_shards = st.multiselect("（未選択=すべて）", shard_ids_all, default=shard_ids_all)

    # 参照ファイル
    st.caption("特定ファイルだけで検索したい場合: 年/ファイル名 をカンマ区切り（例: 2025/foo.pdf, 2024/bar.pdf）")
    file_whitelist_str = st.text_input("参照ファイル（任意）", value="")
    file_whitelist = {s.strip() for s in file_whitelist_str.split(",") if s.strip()}

    # OpenAIキー確認
    has_key = bool(_get_openai_api_key())
    if answer_backend == "OpenAI" and not has_key:
        st.error("OpenAI APIキーが secrets.toml / 環境変数にありません（自動で『Retrieve-only』に切替）。")

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

    # 料金計算（/1K 換算済みの単価を適用）
    st.divider()
    st.subheader("💵 料金計算（/1K tok 換算）")
    usd_jpy = float(st.number_input("為替 (JPY/USD)", min_value=50.0, max_value=500.0,
                                    value=float(DEFAULT_USDJPY), step=0.5))
    chat_in_price = float(MODEL_PRICES_PER_1K.get(chat_model, {}).get("in", 0.0))
    chat_out_price = float(MODEL_PRICES_PER_1K.get(chat_model, {}).get("out", 0.0))
    default_emb_model = "text-embedding-3-large"
    emb_price = float(EMBEDDING_PRICES_USD.get(default_emb_model, 0.0))  # すでに /1K

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

    # 埋め込み backend の安全化
    if embed_backend == "openai" and not _get_openai_api_key():
        st.warning("埋め込みバックエンドが openai ですが APIキー未設定のため、'local' に自動切替します。")
        embed_backend = "local"

    # インライン参照ファイル: [[files: ...]]
    inline = re.search(r"\[\[\s*files\s*:\s*([^\]]+)\]\]", st.session_state.q, flags=re.IGNORECASE)
    inline_files = set()
    if inline:
        inline_files = {s.strip() for s in inline.group(1).split(",") if s.strip()}

    effective_whitelist = {_norm_path(x) for x in (set(file_whitelist) | set(inline_files))}
    clean_q = re.sub(r"\[\[\s*files\s*:[^\]]+\]\]", "", st.session_state.q, flags=re.IGNORECASE).strip()

    # クエリ正規化
    question = normalize_ja_text(clean_q)

    # ベクトルストアのルート
    vs_backend_dir = VS_ROOT / embed_backend
    if not vs_backend_dir.exists():
        st.warning(f"ベクトルが見つかりません（{vs_backend_dir}）。先に **ベクトル化** を同じバックエンドで実行してください。")
        st.stop()

    shard_dirs_all = _list_shard_dirs(embed_backend)
    selected = [vs_backend_dir / s for s in target_shards] if target_shards else [vs_backend_dir / p.name for p in shard_dirs_all]
    shard_dirs = [p for p in selected if p.is_dir() and (p / "vectors.npy").exists()]

    if not shard_dirs:
        st.warning("検索可能なシャードがありません。対象シャードのベクトル化を先に実行してください。")
        st.stop()

    try:
        # --- 検索 ---
        with st.spinner("検索中…"):
            # OpenAI キーを env に注入（EmbeddingStore が env を参照する前提）
            api_key = _get_openai_api_key()
            if api_key and "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = api_key

            estore = EmbeddingStore(backend=embed_backend)
            # OpenAI埋め込み時だけ概算トークン計測（料金の目安）
            emb_tokens = _count_tokens(question, model_hint="text-embedding-3-large") if embed_backend == "openai" else 0
            qv = estore.embed([question]).astype("float32")  # shape=(1, d)

            # 各シャード top-k → 全体マージ（スコア大ほど良い）
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

        # 表示用
        contexts_display = []
        for i, (row_idx, score, meta) in enumerate(raw_hits, 1):
            txt = str(meta.get("text", "") or "")
            src_label = _fmt_source(meta)
            contexts_display.append((i, txt, src_label, float(score)))

        # --- 回答生成 or Retrieve-only ---
        chat_prompt_tokens = 0
        chat_completion_tokens = 0
        answer = None

        use_answer_backend = "Retrieve-only" if (answer_backend == "OpenAI" and not _get_openai_api_key()) else answer_backend

        if use_answer_backend == "OpenAI":
            with st.spinner("回答生成中…"):
                labeled_contexts = [
                    f"[S{i}] {meta.get('text','')}\n[meta: {_fmt_source(meta)} / score={float(score):.3f}]"
                    for i, (_rid, score, meta) in enumerate(raw_hits, 1)
                ]
                # ★ strict=False に変更して、RAG文脈内の情報を積極活用
                prompt = build_prompt(
                    question,
                    labeled_contexts,
                    sys_inst=sys_inst,
                    style_hint=detail,
                    cite=cite,
                    strict=False,
                )

                client = OpenAI(api_key=_get_openai_api_key() or "")

                if _use_responses_api(chat_model):
                    # ---------- Responses API（role分離・max_output_tokens） ----------
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
                        # SDK 差異対策：最小引数で再試行
                        resp = client.responses.create(
                            model=chat_model,
                            input=[
                                {"role": "system", "content": sys_inst},
                                {"role": "user", "content": prompt},
                            ],
                        )

                    # 出力抽出（フォールバック含む）
                    try:
                        answer = resp.output_text
                    except Exception:
                        try:
                            answer = resp.output[0].content[0].text  # 一部SDK系
                        except Exception:
                            answer = str(resp)

                    # 使用トークン
                    try:
                        usage = getattr(resp, "usage", None)
                        if usage:
                            chat_prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                            chat_completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                    except Exception:
                        pass

                else:
                    # ---------- Chat Completions API ----------
                    resp = client.chat.completions.create(
                        model=chat_model,
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                        messages=[
                            {"role": "system", "content": "Follow the user's instruction carefully and answer in Japanese when possible."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    answer = resp.choices[0].message.content
                    try:
                        chat_prompt_tokens = int(resp.usage.prompt_tokens or 0)
                        chat_completion_tokens = int(resp.usage.completion_tokens or 0)
                    except Exception:
                        chat_prompt_tokens = _count_tokens(prompt, model_hint="cl100k_base")
                        chat_completion_tokens = _count_tokens(answer or "", model_hint="cl100k_base")

            st.subheader("🧠 回答")
            st.write(answer)
        else:
            st.subheader("🧩 取得のみ（要約なし）")
            st.info("Retrieve-only モードです。下の参照コンテキストを参照してください。")

        # --- 料金計算（/1K 単価で計算） ---
        with st.container():
            emb_cost_usd = ((emb_tokens / 1000.0) * emb_price) if (embed_backend == "openai") else 0.0
            chat_cost_usd = 0.0
            if use_answer_backend == "OpenAI":
                chat_cost_usd = (chat_prompt_tokens / 1000.0) * chat_in_price + \
                                (chat_completion_tokens / 1000.0) * chat_out_price
            total_usd = emb_cost_usd + chat_cost_usd
            total_jpy = total_usd * usd_jpy

            st.markdown("### 💴 使用料の概算（/1K tok 単価）")
            cols = st.columns(3)
            with cols[0]:
                st.metric("合計 (JPY)", f"{total_jpy:,.2f} 円")
                st.caption(f"為替 {usd_jpy:.2f} JPY/USD")
            with cols[1]:
                st.write("**内訳 (USD)**")
                st.write(f"- Embedding: `${emb_cost_usd:.6f}` ({emb_tokens} tok @ {emb_price:.5f}/1K)")
                if use_answer_backend == "OpenAI":
                    st.write(f"- Chat 入力: `${(chat_prompt_tokens/1000.0)*chat_in_price:.6f}` ({chat_prompt_tokens} tok @ {chat_in_price:.5f}/1K)")
                    st.write(f"- Chat 出力: `${(chat_completion_tokens/1000.0)*chat_out_price:.6f}` ({chat_completion_tokens} tok @ {chat_out_price:.5f}/1K)")
                st.write(f"- 合計: `${total_usd:.6f}`")
            with cols[2]:
                st.write("**単価 (USD / 1K tok)**")
                st.write(f"- Embedding: `${emb_price:.5f}`（{default_emb_model}）")
                st.write(f"- Chat 入力: `${chat_in_price:.5f}`（{chat_model}）")
                st.write(f"- Chat 出力: `${chat_out_price:.5f}`（{chat_model}）")

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
