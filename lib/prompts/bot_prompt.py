# lib/prompts/bot_prompt.py
# ============================================
# 社内ボット（RAG）の厳格プロンプトを生成するユーティリティ
# - “資料外は分かりません” を強制（strict mode）
# - スタイル（style hint）のプリセット付き
# - 引用表記（[S1], [S2] ...）の明示を指示可能
# ============================================

from __future__ import annotations
from typing import List

DEFAULT_SYS_INST = "あなたは社内アシスタントです。"

# スタイルのプリセット（style hints）
STYLE_MAP = {
    "concise": "箇条書きで要点のみ、150-250字程度。",
    "standard": "見出し＋箇条書きで300-500字程度。",
    "detailed": "見出し＋箇条書き＋要約で500-800字程度。",
    "very_detailed": "丁寧な要約と段落構成で800字以上。",
}

def _guard_text(strict: bool) -> str:
    """
    strict=True のとき：
      コンテキスト（retrieved contexts）以外の知識は禁止。
      “この資料からは分かりません” だけを返す／
      提案や一般論の補足もしない（no suggestion, no speculation）。
    """
    if strict:
        return (
            "以下のコンテキストに書かれていること【のみ】を根拠に回答してください。"
            "質問に対応する情報がコンテキストに含まれていない場合、"
            "『この資料からは分かりません』とだけ答えてください。"
            "絶対に一般知識・推測・提案・補足を加えないでください。"
        )
    else:
        return "必要に応じて一般知識で補足しても構いません。"

def build_prompt(
    question: str,
    labeled_contexts: List[str],
    *,
    sys_inst: str = DEFAULT_SYS_INST,
    style_hint: str = "standard",
    cite: bool = True,
    strict: bool = True,
) -> str:
    """
    RAG 回答用のプロンプト（prompt）を生成します。

    Parameters
    ----------
    question : str
        ユーザー質問（user question）
    labeled_contexts : List[str]
        取得済み文脈（retrieved contexts）。
        例: ["[S1] 本文...\n[meta: 2025/a.pdf p.3 / score=0.812]", "[S2] ...", ...]
    sys_inst : str, default DEFAULT_SYS_INST
        system role の指示（system instruction）
    style_hint : {"concise","standard","detailed","very_detailed"}, default "standard"
        出力の粒度・分量（style）
    cite : bool, default True
        回答内に [S1], [S2] などの参照明記を要求（citation line を挿入）
    strict : bool, default True
        厳格モード。コンテキスト外の情報は禁止（no suggestion / no speculation）

    Returns
    -------
    str
        モデルへ渡す最終プロンプト（multi-section text）
    """
    style = STYLE_MAP.get(style_hint, STYLE_MAP["standard"])
    ctx = "\n\n".join(labeled_contexts) if labeled_contexts else "(なし)"
    citeline = "根拠箇所は [S1], [S2] の形式で必ず明記してください。" if cite else ""
    guard = _guard_text(strict)

    prompt = f"""# System
{sys_inst}

# Task
次のユーザー質問に、日本語で回答してください。

# User Question
{question}

# Retrieved Contexts
{ctx}

# Requirements
- {style}
- {citeline}
- {guard}
"""
    return prompt
