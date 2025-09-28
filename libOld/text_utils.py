# lib/text_utils.py
from __future__ import annotations
import re
import unicodedata

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

def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")

def make_snippet(text: str, pats, total_len: int = 240) -> str:
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
