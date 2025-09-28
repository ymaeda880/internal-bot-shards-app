# lib/openai_utils.py
from __future__ import annotations
from typing import List
import re
from openai import OpenAI

# ---- token helpers ----
def _encoding_for(model_hint: str):
    import tiktoken
    try:
        return tiktoken.encoding_for_model(model_hint)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model_hint: str = "gpt-5-mini") -> int:
    try:
        enc = _encoding_for(model_hint)
        return len(enc.encode(text or ""))
    except Exception:
        return max(1, int(len(text or "") / 4))

def truncate_by_tokens(text: str, max_tokens: int, model_hint: str = "gpt-5-mini") -> str:
    try:
        enc = _encoding_for(model_hint)
        toks = enc.encode(text or "")
        if len(toks) <= max_tokens:
            return text or ""
        return enc.decode(toks[:max_tokens])
    except Exception:
        max_chars = max(100, max_tokens * 4)
        return (text or "")[:max_chars]

def is_gpt5(model_name: str) -> bool:
    return (model_name or "").lower().startswith("gpt-5")

# ---------- OpenAI 呼び出し ----------
def chat_complete_safely(client: OpenAI, *, model: str, temperature: float,
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
    prefer_mct = (model.lower().startswith("gpt-5") or model.lower().startswith("o"))
    try:
        return _call(prefer_mct)
    except Exception:
        return _call(not prefer_mct)

def extract_text_from_chat(resp_obj) -> str:
    try:
        content = resp_obj.choices[0].message.content
        return content or ""
    except Exception:
        return ""

def responses_generate(client: OpenAI, *, model: str, temperature: float,
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

def responses_text(resp) -> str:
    try:
        txt = resp.output_text
        if isinstance(txt, str) and txt.strip():
            return txt
    except Exception:
        pass
    return ""
