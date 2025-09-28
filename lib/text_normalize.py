# lib/text_normalize.py
from __future__ import annotations
import unicodedata
import re

# ========= 正規表現パターン =========
CJK = r"\u3040-\u309F\u30A0-\u30FF\uFF65-\uFF9F\u4E00-\u9FFF\u3400-\u4DBF"
PUNC = r"、。・，．！？：；（）［］｛｝「」『』〈〉《》【】"

_cjk_cjk_space = re.compile(fr"(?<=[{CJK}])\s+(?=[{CJK}])")
_space_before_punc = re.compile(fr"\s+(?=[{PUNC}])")
_space_after_open = re.compile(fr"(?<=[（［｛「『〈《【])\s+")
_space_before_close = re.compile(fr"\s+(?=[）］｝」』〉》】])")
_multi_space = re.compile(r"[ \t\u3000]+")

# ========= 関数 =========
def normalize_ja_text(s: str) -> str:
    """
    日本語テキストを正規化する関数
    - NFKC 正規化
    - 全角・半角スペースを整理
    - CJK 文字間の不要な空白削除
    - 句読点まわりのスペース削除
    """
    s = unicodedata.normalize("NFKC", s)
    s = _cjk_cjk_space.sub("", s)
    s = _space_before_punc.sub("", s)
    s = _space_after_open.sub("", s)
    s = _space_before_close.sub("", s)
    s = _multi_space.sub(" ", s)
    return s.strip()
