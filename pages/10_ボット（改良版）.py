# ============================================
# 変更点（この版での修正・追加）
# --------------------------------------------
# 1) OpenAIキー未設定時の安全化:
#    - 埋め込み backend=openai かつ OPENAI_API_KEY 未設定なら自動で 'local' に切替。
# 2) vdb.search() のスコア向きを明示:
#    - return_="similarity" を常に指定（「大きいほど良い」前提でヒープ/ソート）。
# 3) 参照ファイルホワイトリストの安全比較:
#    - NFKC/大小文字/区切り(\\ → /) 正規化して一致判定（_norm_path）。
# 4) インライン [[files: ...]] 抽出/除去の堅牢化:
#    - 余白許容・大文字小文字無視の正規表現に強化。
# 5) send_now 判定の安全化:
#    - go = go_click or bool(locals().get("send_now"))
# 6) コメントで「score は similarity」を明示し将来の混在を抑止。
# ============================================

# pages/09_ボット.py
# ------------------------------------------------------------
# 💬 Internal Bot (RAG, No-FAISS) — 以前のボット風UI + シャード横断検索版（参照ファイル指定対応）
# - data/vectorstore/<backend>/<shard_id>/ を横断して top-k マージ
# - 日本語の「1文字ごと空白」問題に対応（normalize_ja_text）
# - OpenAI キー未設定時は自動で Retrieve-only に切替
# - vdb.search の戻り値が (row_idx, score, meta) / (score, meta) どちらでも受ける
# - 参照ファイルをサイドバー／質問内 [[files: 2025/a.pdf, 2024/b.pdf]] で指定可能
# ------------------------------------------------------------

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import random
import heapq
import unicodedata
import re

import streamlit as st
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from rag_utils import EmbeddingStore, NumpyVectorDB, build_prompt

# ========= パス =========
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
VS_ROOT  = DATA_DIR / "vectorstore"

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

def _format_contexts_for_prompt(hits: List[Tuple[int, float, Dict[str, Any]]]) -> List[str]:
    labeled = []
    for i, (idx, score, meta) in enumerate(hits, 1):
        txt = str(meta.get("text", "") or "")
        file_label = _fmt_source(meta)
        labeled.append(f"[S{i}] {txt}\n[meta: {file_label} / score={float(score):.3f}]")
    return labeled

def _list_shard_dirs(backend: str) -> List[Path]:
    base = VS_ROOT / backend
    if not base.exists(): return []
    return sorted([p for p in base.iterdir() if p.is_dir()])

def _norm_path(s: str) -> str:
    """年/ファイル名の一致を安定させるための正規化"""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().replace("\\", "/")
    return s.lower()

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
# Streamlit 準備（以前のボット風UI）
# =========================
load_dotenv()
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

    # 検索件数
    top_k = st.slider("検索件数（Top-K）", 1, 12, 6, 1)

    # 回答スタイル
    label_to_value = {"簡潔":"concise","標準":"standard","詳細":"detailed","超詳細":"very_detailed"}
    detail_label = st.selectbox("詳しさ", list(label_to_value.keys()), index=2)
    detail = label_to_value[detail_label]

    cite = st.checkbox("出典を角括弧で引用（[S1] 等）", value=True)
    max_tokens = st.slider("最大トークン数", 256, 4000, 1200, 64)
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)

    answer_backend = st.radio("回答モデル", ["OpenAI", "Retrieve-only"], index=0)
    sys_inst = st.text_area("System Instruction", "あなたは優秀な社内のアシスタントです.", height=80)

    # シャード選択
    st.divider()
    st.subheader("検索対象シャード")
    shard_dirs_all = _list_shard_dirs(embed_backend)
    shard_ids_all = [p.name for p in shard_dirs_all]
    target_shards = st.multiselect("（未選択=すべて）", shard_ids_all, default=shard_ids_all)

    # --- 参照ファイル（任意）: 例 "2025/foo.pdf, 2024/bar.pdf"
    st.caption("特定ファイルだけで検索したい場合は、年/ファイル名 でカンマ区切り指定（例: 2025/foo.pdf, 2024/bar.pdf）")
    file_whitelist_str = st.text_input("参照ファイル（任意）", value="")
    file_whitelist = {s.strip() for s in file_whitelist_str.split(",") if s.strip()}

    # OpenAI キーが無いならここで警告表示
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if answer_backend == "OpenAI" and not has_key:
        st.error("OPENAI_API_KEY が .env にありません（自動で『Retrieve-only』に切り替えます）。")

    st.divider()
    # サンプル質問
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

    st.divider()
    # 料金計算
    st.subheader("💵 料金計算（編集可）")
    fx_rate = st.number_input("為替レート (JPY/USD)", min_value=50.0, max_value=500.0, value=150.0, step=0.5)
    chat_in_price = st.number_input("Chat 入力単価 (USD / 1K tok)", min_value=0.0, value=0.00015, step=0.00001, format="%.5f")
    chat_out_price = st.number_input("Chat 出力単価 (USD / 1K tok)", min_value=0.0, value=0.00060, step=0.00001, format="%.5f")
    emb_price = st.number_input("Embedding 単価 (USD / 1K tok)", min_value=0.0, value=0.00002, step=0.00001, format="%.5f")

# 入力欄
q = st.text_area("質問を入力", value=st.session_state.q, placeholder="この社内ボットに質問してください…", height=100)
if q != st.session_state.q:
    st.session_state.q = q

go_click = st.button("送信", type="primary")
go = go_click or bool(locals().get("send_now"))

# =========================
# 実行
# =========================
if go and st.session_state.q.strip():

    # --- 埋め込み backend の安全化（OpenAIキー未設定なら local に切替）
    if embed_backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.warning("埋め込みバックエンドが openai ですが OPENAI_API_KEY が未設定のため、'local' に自動切替します。")
        embed_backend = "local"

    # --- インライン指定の取り出し: [[files: 2025/aaa.pdf, 2024/bbb.pdf]]
    inline = re.search(r"\[\[\s*files\s*:\s*([^\]]+)\]\]", st.session_state.q, flags=re.IGNORECASE)
    inline_files = set()
    if inline:
        inline_files = {s.strip() for s in inline.group(1).split(",") if s.strip()}

    # UI入力と統合（どちらか/両方OK）→ 正規化
    effective_whitelist_raw = set(file_whitelist) | set(inline_files)
    effective_whitelist = {_norm_path(x) for x in effective_whitelist_raw}

    # インラインタグは本文から除去 → 正規化へ
    clean_q = re.sub(r"\[\[\s*files\s*:[^\]]+\]\]", "", st.session_state.q, flags=re.IGNORECASE).strip()

    # ★ クエリ正規化（日本語の1字空白対策）
    question = normalize_ja_text(clean_q)

    # 使うベクトルストアのルート（シャード横断）
    vs_backend_dir = VS_ROOT / embed_backend
    if not vs_backend_dir.exists():
        st.warning(f"ベクトルが見つかりません（{vs_backend_dir}）。先に **ベクトル化** を同じバックエンドで実行してください。")
        st.stop()

    shard_dirs_all = _list_shard_dirs(embed_backend)  # 念のため再取得（実行時変更対策）
    selected = [vs_backend_dir / s for s in target_shards] if target_shards else [vs_backend_dir / p.name for p in shard_dirs_all]
    shard_dirs = [p for p in selected if p.is_dir() and (p / "vectors.npy").exists()]

    if not shard_dirs:
        st.warning("検索可能なシャードがありません。対象シャードのベクトル化を先に実行してください。")
        st.stop()

    try:
        # --- 検索 ---
        with st.spinner("検索中…"):
            estore = EmbeddingStore(backend=embed_backend)
            emb_tokens = _count_tokens(question, model_hint="text-embedding-3-small") if embed_backend == "openai" else 0
            qv = estore.embed([question]).astype("float32")  # shape=(1, d)

            # 各シャードで top_k を取り、全体でマージ（最小ヒープ）
            K = int(top_k)
            heap: List[Tuple[float, Tuple[int, Dict[str, Any]]]] = []  # (score, (row_idx, meta))

            for shp in shard_dirs:
                try:
                    vdb = NumpyVectorDB(shp)  # metric の既定は rag_utils 側に依存
                    # 類似度（大きいほど良い）を返す契約で取得（将来の距離実装と混在しないよう明示）
                    hits = vdb.search(qv, top_k=K, return_="similarity")
                    for h in hits:
                        # 戻り値の揺れに対応
                        if isinstance(h, tuple) and len(h) == 3:
                            row_idx, score, meta = h
                        elif isinstance(h, tuple) and len(h) == 2:
                            score, meta = h
                            row_idx = -1
                        else:
                            continue

                        md = dict(meta or {})
                        md["shard_id"] = shp.name

                        # ▼ 参照ファイル指定がある場合はここでフィルタ（正規化後の完全一致）
                        if effective_whitelist:
                            if _norm_path(str(md.get("file", ""))) not in effective_whitelist:
                                continue

                        sc = float(score)  # similarity（大きいほど良い）前提

                        if len(heap) < K:
                            heapq.heappush(heap, (sc, (row_idx, md)))
                        else:
                            if sc > heap[0][0]:
                                heapq.heapreplace(heap, (sc, (row_idx, md)))
                except Exception as e:
                    st.warning(f"シャード {shp.name} の検索でエラー: {e}")

            raw_hits = [(rid, sc, md) for sc, (rid, md) in sorted(heap, key=lambda x: x[0], reverse=True)]

        if not raw_hits:
            if effective_whitelist:
                st.warning("指定された参照ファイル内で該当コンテキストが見つかりませんでした。"
                           "ファイル名と年（例: 2025/foo.pdf）をご確認ください。")
            else:
                st.warning("該当コンテキストが見つかりませんでした。チャンクサイズや Top-K を調整して再試行してください。")
            st.stop()

        # 画面表示用
        contexts_display = []
        for i, (row_idx, score, meta) in enumerate(raw_hits, 1):
            txt = str(meta.get("text", "") or "")
            src_label = _fmt_source(meta)
            contexts_display.append((i, txt, src_label, float(score)))

        # --- 回答生成 or Retrieve-only ---
        chat_prompt_tokens = 0
        chat_completion_tokens = 0
        answer = None

        # OpenAI キーが無いなら自動で Retrieve-only に切替
        use_answer_backend = "Retrieve-only" if (answer_backend == "OpenAI" and not os.getenv("OPENAI_API_KEY")) else answer_backend

        if use_answer_backend == "OpenAI":
            with st.spinner("回答生成中…"):
                labeled_contexts = _format_contexts_for_prompt(raw_hits)
                prompt = build_prompt(
                    question,
                    labeled_contexts,
                    sys_inst=sys_inst,
                    style_hint=detail,
                    cite=cite,
                    strict=True,
                )
                client = OpenAI()
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
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
                    chat_prompt_tokens = _count_tokens(prompt, model_hint="gpt-4o-mini")
                    chat_completion_tokens = _count_tokens(answer or "", model_hint="gpt-4o-mini")

            st.subheader("🧠 回答")
            st.write(answer)
        else:
            st.subheader("🧩 取得のみ（要約なし）")
            st.info("Retrieve-only モードです。下の参照コンテキストを参照してください。")

        # --- 料金計算（概算） ---
        with st.container():
            emb_cost_usd = (emb_tokens / 1000.0) * float(emb_price) if embed_backend == "openai" else 0.0
            chat_cost_usd = 0.0
            if use_answer_backend == "OpenAI":
                chat_cost_usd = (chat_prompt_tokens / 1000.0) * float(chat_in_price) + \
                                (chat_completion_tokens / 1000.0) * float(chat_out_price)
            total_usd = emb_cost_usd + chat_cost_usd
            total_jpy = total_usd * float(fx_rate)

            st.markdown("### 💴 使用料の概算")
            cols = st.columns(3)
            with cols[0]:
                st.metric("合計 (JPY)", f"{total_jpy:,.2f} 円")
                st.caption(f"為替 {float(fx_rate):.2f} JPY/USD")
            with cols[1]:
                st.write("**内訳 (USD)**")
                st.write(f"- Embedding: `${emb_cost_usd:.6f}` ({emb_tokens} tok)")
                if use_answer_backend == "OpenAI":
                    st.write(f"- Chat 入力: `${(chat_prompt_tokens/1000.0)*float(chat_in_price):.6f}` ({chat_prompt_tokens} tok)")
                    st.write(f"- Chat 出力: `${(chat_completion_tokens/1000.0)*float(chat_out_price):.6f}` ({chat_completion_tokens} tok)")
                st.write(f"- 合計: `${total_usd:.6f}`")
            with cols[2]:
                st.write("**単価 (USD / 1K tok)**")
                st.write(f"- Embedding: `${float(emb_price):.5f}`")
                st.write(f"- Chat 入力: `${float(chat_in_price):.5f}`")
                st.write(f"- Chat 出力: `${float(chat_out_price):.5f}`")

        # --- 参照コンテキスト（折りたたみ） ---
        with st.expander("🔍 参照コンテキスト（上位ヒット）", expanded=False):
            for i, txt, src_label, score in contexts_display:
                snippet = (txt[:1000] + "…") if len(txt) > 1000 else txt  # 体感軽量化
                st.markdown(f"**[S{i}] score={score:.3f}**  `{src_label}`\n\n{snippet}")

    except Exception as e:
        st.error(f"検索/生成中にエラー: {e}")
else:
    st.info("質問を入力して『送信』を押してください。サイドバーでシャードや回答設定、参照ファイルを調整できます。")
