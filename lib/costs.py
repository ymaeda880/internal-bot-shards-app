# lib/costs.py
from __future__ import annotations
from dataclasses import dataclass
import streamlit as st

# ============================================
# 為替の初期値（secretsにUSDJPYがあれば上書き）
# ============================================
DEFAULT_USDJPY = float(st.secrets.get("USDJPY", 150.0))

# ============================================
# モデル価格（USD / 100万トークン）
# ============================================
MODEL_PRICES_USD = {
    "gpt-5":         {"in": 1.25,  "out": 10.00},
    "gpt-5-mini":    {"in": 0.25,  "out": 2.00},
    "gpt-5-nano":    {"in": 0.05,  "out": 0.40},
    "gpt-4o":        {"in": 2.50,  "out": 10.00},
    "gpt-4o-mini":   {"in": 0.15,  "out": 0.60},
    "gpt-4.1":       {"in": 2.00,  "out": 8.00},   # 参考
    "gpt-4.1-mini":  {"in": 0.40,  "out": 1.60},   # 参考
    "gpt-3.5-turbo": {"in": 0.50,  "out": 1.50},   # 参考
}

# ============================================
# Embedding 価格（USD / 100万トークン）
# ============================================
EMBEDDING_PRICES_USD = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,  # レガシー
}

# ============================================
# 音声（Whisper）価格（USD / 分）
# ============================================
AUDIO_PRICES_USD_PER_MIN = {
    "whisper-1": 0.006,   # $0.006 / 分
}

MILLION = 1_000_000

@dataclass
class ChatUsage:
    input_tokens: int
    output_tokens: int

# 共通：USD→JPY 変換
def usd_to_jpy(usd: float, rate: float = DEFAULT_USDJPY) -> float:
    return round(usd * rate, 2)

def estimate_chat_cost(model: str, usage: ChatUsage) -> dict:
    if model not in MODEL_PRICES_USD:
        raise ValueError(f"単価未設定のモデル: {model}")

    price = MODEL_PRICES_USD[model]
    in_cost  = (usage.input_tokens  / MILLION) * price["in"]
    out_cost = (usage.output_tokens / MILLION) * price["out"]
    usd = round(in_cost + out_cost, 6)
    jpy = usd_to_jpy(usd)
    return {"usd": usd, "jpy": jpy}

def estimate_embedding_cost(model: str, input_tokens: int) -> dict:
    if model not in EMBEDDING_PRICES_USD:
        raise ValueError(f"単価未設定の埋め込みモデル: {model}")
    usd = round((input_tokens / MILLION) * EMBEDDING_PRICES_USD[model], 6)
    jpy = usd_to_jpy(usd)
    return {"usd": usd, "jpy": jpy}

def estimate_transcribe_cost(model: str, seconds: float) -> dict:
    if model not in AUDIO_PRICES_USD_PER_MIN:
        raise ValueError(f"単価未設定の音声モデル: {model}")
    per_min = AUDIO_PRICES_USD_PER_MIN[model]
    minutes = max(0.0, seconds) / 60.0
    usd = round(per_min * minutes, 6)
    jpy = usd_to_jpy(usd)
    return {"usd": usd, "jpy": jpy}
