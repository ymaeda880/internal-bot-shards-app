# lib/search_engine.py
from typing import List, Dict
import pandas as pd
import streamlit as st
from pathlib import Path
import re

from lib.text_utils import normalize_ja_text, make_snippet
from lib.search_utils import iter_jsonl

def compile_terms(q: str, use_regex: bool, case_sensitive: bool, normalize_query: bool):
    if normalize_query:
        q = normalize_ja_text(q)
    terms = [t for t in q.split() if t]
    if not terms:
        return []
    flags = 0 if case_sensitive else re.IGNORECASE
    pats = []
    for t in terms:
        try:
            pats.append(re.compile(t if use_regex else re.escape(t), flags))
        except re.error:
            pats.append(re.compile(re.escape(t), flags))
    return pats

def run_search(sel_shards, base_dir: Path, query: str,
               year_min: int, year_max: int, file_filter: str,
               max_rows: int, snippet_len: int,
               normalize_query: bool, norm_body: bool,
               bool_mode: str, use_regex: bool, case_sensitive: bool,
               show_cols: List[str]) -> pd.DataFrame | None:
    pats = compile_terms(query, use_regex, case_sensitive, normalize_query)
    if not pats:
        st.warning("検索語が空です。")
        return None

    rows: List[Dict] = []
    total_scanned = 0
    for sid in sel_shards:
        meta_path = base_dir / sid / "meta.jsonl"
        for obj in iter_jsonl(meta_path):
            total_scanned += 1
            yr = obj.get("year")
            if isinstance(yr, int):
                if year_min and yr < year_min: continue
                if year_max and year_max < 9999 and yr > year_max: continue
            if file_filter and file_filter.lower() not in str(obj.get("file","")).lower():
                continue
            text = str(obj.get("text",""))
            text_for_match = normalize_ja_text(text) if norm_body else text
            ok = all(p.search(text_for_match) for p in pats) if bool_mode=="AND" \
                 else any(p.search(text_for_match) for p in pats)
            if not ok: continue
            score = sum(len(list(p.finditer(text_for_match))) for p in pats)
            rows.append({
                "file": obj.get("file"), "year": obj.get("year"),
                "page": obj.get("page"), "shard_id": sid,
                "score": score, "text": make_snippet(text, pats, snippet_len),
            })
            if len(rows) >= max_rows: break
        if len(rows) >= max_rows: break

    if not rows:
        st.warning("ヒットなし。")
        return None

    df = pd.DataFrame(rows).sort_values(["score","year","file","page"],
                                        ascending=[False, True, True, True])
    st.success(f"ヒット {len(df)} 件 / 走査 {total_scanned} レコード")
    return df
