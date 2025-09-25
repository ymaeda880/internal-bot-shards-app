import streamlit as st

# OpenAI API
OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

def get_openai_api_key() -> str:
    return st.secrets.get("OPENAI_API_KEY", "")

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
    "text-embedding-3-small": 0.02,   # $0.00002 / 1K tok
    "text-embedding-3-large": 0.13,   # $0.00013 / 1K tok
    "text-embedding-ada-002": 0.10,   # レガシー参考
}

def estimate_embedding_cost_usd(tokens: int, model: str) -> float:
    per_million = EMBEDDING_PRICES_USD.get(model)
    if per_million is None:
        raise ValueError(f"Unknown embedding model: {model}")
    return (tokens / 1_000_000.0) * per_million

def estimate_embedding_cost_jpy(tokens: int, model: str, usd_jpy: float | None = None) -> float:
    r = usd_jpy if usd_jpy is not None else DEFAULT_USDJPY
    return estimate_embedding_cost_usd(tokens, model) * float(r)

# ============================================
# Whisper（USD / 分）
# ============================================
WHISPER_PRICE_PER_MIN = 0.006

# ============================================
# 為替の初期値（secretsにUSDJPYがあれば上書き）
# ============================================
DEFAULT_USDJPY = float(st.secrets.get("USDJPY", 150.0))

# ============================================
# モデル別の推奨出力上限（目安）
# ============================================
v1 = 128000
MAX_COMPLETION_BY_MODEL = {
    "gpt-5": v1,
    "gpt-5-mini": v1,
    "gpt-5-nano": v1,
    "gpt-4.1": v1,
    "gpt-4.1-mini": v1,
    "gpt-4o": v1,
    "gpt-4o-mini": v1,
    "gpt-3.5-turbo": 10000,
}
