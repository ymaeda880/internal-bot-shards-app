# rag_utiles.py
#
# ============================================
# 修正点（前版からの変更）
# --------------------------------------------
# - np.argpartition の off-by-one を修正（top_k → top_k-1）
# - score の向きを明示化：search(..., return_="similarity"|"distance")
# - metric="cosine"|"dot"|"euclidean" を扱えるようにし、返り値は既定で「大きいほど良い」に統一
# - メタ件数 < ベクトル件数 の場合も安全に空dictを返すよう修正
# - build_prompt / load_docx_files を __all__ に追加
# - 日本語正規化ユーティリティ normalize_ja_text を追加
# ============================================

import os
import json
import importlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from docx import Document

import numpy as np
import unicodedata
import re

# ---- 安定化（任意）----
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---------- 日本語テキスト正規化（任意で使う） ----------
CJK = r"\u3040-\u309F\u30A0-\u30FF\uFF65-\uFF9F\u4E00-\u9FFF\u3400-\u4DBF"
PUNC = r"、。・，．！？：；（）［］｛｝「」『』〈〉《》【】"
_rx_cjk_space   = re.compile(fr"(?<=[{CJK}])\s+(?=[{CJK}])")
_rx_before_punc = re.compile(fr"\s+(?=[{PUNC}])")
_rx_after_open  = re.compile(fr"(?<=[（［｛「『〈《【])\s+")
_rx_before_close= re.compile(fr"\s+(?=[）］｝」』〉》】])")
_rx_multi_space = re.compile(r"[ \t\u3000]+")

def normalize_ja_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = _rx_cjk_space.sub("", s)
    s = _rx_before_punc.sub("", s)
    s = _rx_after_open.sub("", s)
    s = _rx_before_close.sub("", s)
    s = _rx_multi_space.sub(" ", s)
    return s.strip()

# ---------- テキスト分割 ----------
def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[Tuple[str, int, int]]:
    """
    テキストをチャンク分割し、(チャンク本文, span_start, span_end) のリストを返す
    """
    chunks = []
    n = len(text)
    i = 0
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append((chunk, i, j))
        if j >= n:
            break
        i = max(0, j - overlap)
    return chunks

# ---------- .txt ローダ ----------
def load_txt_files(data_dir: Path) -> List[Tuple[str, str]]:
    pairs = []
    data_dir.mkdir(exist_ok=True)
    for p in sorted(data_dir.glob("*.txt")):
        try:
            txt = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = p.read_text(encoding="cp932", errors="ignore")
        pairs.append((p.name, txt))
    return pairs

# ---------- 埋め込み ----------
class EmbeddingStore:
    """
    backend='local' : sentence-transformers/all-MiniLM-L6-v2（384次元）
    backend='openai': text-embedding-3-small（1536次元）
    いずれも cosine 用に L2 正規化して返します。
    """
    def __init__(self, backend: str = "local",
                 local_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 openai_model: str = "text-embedding-3-small"):
        self.backend = backend
        self.local_model = local_model
        self.openai_model = openai_model
        self._model = None
        self._client = None

    def _ensure_local(self):
        if self._model is None:
            st_mod = importlib.import_module("sentence_transformers")
            SentenceTransformer = getattr(st_mod, "SentenceTransformer")
            self._model = SentenceTransformer(self.local_model)

    def _ensure_openai(self):
        if self._client is None:
            oa_mod = importlib.import_module("openai")
            OpenAI = getattr(oa_mod, "OpenAI")
            self._client = OpenAI()

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        if self.backend == "openai":
            self._ensure_openai()
            out = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                resp = self._client.embeddings.create(model=self.openai_model, input=batch)
                vecs = [np.array(d.embedding, dtype="float32") for d in resp.data]
                # cosine 用に正規化
                vecs = [v / (np.linalg.norm(v) + 1e-12) for v in vecs]
                out.extend(vecs)
            # 返却は L2 正規化済み
            return np.vstack(out) if out else np.zeros((0, 1536), dtype="float32")

        # local
        self._ensure_local()
        embs = self._model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        return embs  # 返却は L2 正規化済み

# ---------- 超シンプル VectorDB（NumPyのみ） ----------
class NumpyVectorDB:
    """
    vectors.npy: shape (N, d) float32（正規化済を想定）
    meta.jsonl : 1行1 JSON {"file":..., "page":..., "chunk_id":..., "chunk_index":..., "text":..., "span_start":..., "span_end":...}
    """
    def __init__(self, base_dir: Path, metric: str = "cosine"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.base_dir / "vectors.npy"
        self.meta_path = self.base_dir / "meta.jsonl"
        self.metric = metric  # "cosine" | "dot" | "euclidean"

    def reset(self):
        if self.vec_path.exists():
            self.vec_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()

    def _load_vectors(self, mmap: bool = True) -> Optional[np.ndarray]:
        if not self.vec_path.exists():
            return None
        # numpy の mmap_mode は 'r' / 'r+' / 'w+' / 'c' または None
        return np.load(self.vec_path, mmap_mode=('r' if mmap else None))

    def _load_meta(self) -> List[dict]:
        if not self.meta_path.exists():
            return []
        out = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def _append_meta(self, items: List[dict]):
        with open(self.meta_path, "a", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    def add(self, embeddings: np.ndarray, meta_items: List[dict]):
        """
        embeddings: shape (N, d)（L2 正規化済みを想定）
        meta_items: [{"file":..., "page":..., "chunk_id":..., "chunk_index":..., "text":..., "span_start":..., "span_end":...}, ...]
        """
        assert embeddings.ndim == 2 and len(meta_items) == embeddings.shape[0]

        if self.vec_path.exists():
            old = np.load(self.vec_path)  # ここは vstack のためにロード（mmap不可）
            if old.shape[1] != embeddings.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: existing {old.shape[1]} vs new {embeddings.shape[1]}"
                )
            new = np.vstack([old, embeddings])
        else:
            new = embeddings
        np.save(self.vec_path, new)

        self._append_meta(meta_items)

    def _similarity(self, q: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        返り値は「大きいほど良い」に統一して返す。
        """
        if self.metric == "cosine":
            # どちらも正規化済み前提 → 内積が cosine similarity
            return (X @ q).astype("float32")
        elif self.metric == "dot":
            return (X @ q).astype("float32")  # 正規化なし前提ならスケール注意
        elif self.metric == "euclidean":
            # 距離 d を similarity = -d に変換（大きいほど良い）
            d = np.linalg.norm(X - q, axis=1)
            return (-d).astype("float32")
        else:
            raise ValueError(f"unknown metric: {self.metric}")

    def search(self, query_vec: np.ndarray, top_k: int = 5, return_: str = "similarity"):
        """
        return_: "similarity"（既定）= 大きいほど良い, "distance" = 小さいほど良い を明示的に選択可能。
        metric に応じて自動変換。
        """
        vecs = self._load_vectors(mmap=True)
        if vecs is None or vecs.shape[0] == 0:
            return []
        if query_vec.ndim != 2 or query_vec.shape[0] != 1:
            raise ValueError("query_vec は shape=(1, d) を想定しています。")
        if vecs.shape[1] != query_vec.shape[1]:
            raise ValueError("次元数が一致しません。")

        q = query_vec[0]
        scores_sim = self._similarity(q, vecs)  # 大きいほど良い

        # 上位Kインデックス（降順）。np.argpartition の off-by-one を回避
        n = len(scores_sim)
        if top_k >= n:
            top_idx = np.argsort(-scores_sim)
        else:
            part = np.argpartition(-scores_sim, top_k - 1)[:top_k]
            top_idx = part[np.argsort(-scores_sim[part])]

        metas = self._load_meta()
        out = []
        for i in top_idx[:top_k]:
            meta = metas[i] if i < len(metas) else {}
            if return_ == "similarity":
                score = float(scores_sim[i])
                out.append((int(i), score, meta))
            elif return_ == "distance":
                # similarity → distance に変換して返す
                if self.metric == "euclidean":
                    dist = float(-scores_sim[i])  # 復元
                elif self.metric == "cosine":
                    dist = float(1.0 - scores_sim[i])  # cosine distance
                elif self.metric == "dot":
                    dist = float(-scores_sim[i])  # 参考的定義（非推奨）
                else:
                    dist = float(-scores_sim[i])
                out.append((int(i), dist, meta))
            else:
                raise ValueError("return_ must be 'similarity' or 'distance'")
        return out

# ---------- 処理済みファイル管理 ----------
class ProcessedFilesSimple:
    """処理済みファイル名だけを保存・照会するシンプルなマニフェスト"""
    def __init__(self, json_path: Path):
        self.path = json_path
        self.names = set()
        if self.path.exists():
            try:
                self.names = set(json.loads(self.path.read_text(encoding="utf-8")))
            except Exception:
                self.names = set()

    def is_done(self, name: str) -> bool:
        return name in self.names

    def mark_done(self, name: str):
        self.names.add(name)
        self.path.write_text(
            json.dumps(sorted(self.names), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def reset(self):
        self.names.clear()
        if self.path.exists():
            self.path.unlink()

# ---------- 回答用プロンプト構築 ----------
def build_prompt(
    question: str,
    contexts: List[str],
    sys_inst: str = "あなたは優秀な社内アシスタントです。",
    style_hint: str = "standard",
    cite: bool = True,
    strict: bool = True,
) -> str:
    """
    Context をもとに回答プロンプトを組み立てるユーティリティ関数。
    - question: ユーザーの質問
    - contexts: 検索でヒットしたテキスト（複数、[S1] ... の形式）
    - sys_inst: システムインストラクション
    - style_hint: 回答スタイル（concise / standard / detailed / very_detailed）
    - cite: 出典を強制するか
    - strict: Context外の知識を禁止するか
    """
    style_notes = {
        "concise": "要点のみ簡潔にまとめてください（150〜250語程度）。",
        "standard": "分かりやすく構造化して回答してください。",
        "detailed": "要約→詳細→手順→例→注意点の順で説明してください。",
        "very_detailed": "包括的な手順書として、例・エッジケース・チェックリストも含めて詳述してください。",
    }
    requirements = [
        "以下のContextの記述だけを根拠に回答してください。一般知識・推測は禁止。",
        style_notes.get(style_hint, style_notes["standard"]),
        "回答は日本語で記述してください。",
    ]
    if cite:
        requirements.append("引用した事実には直後に [S1] のように出典IDを必ず付けてください。")
    if strict:
        requirements.append("Contextに該当が無い場合は『分かりません』と答えてください。")
    ctx_text = "\n\n".join(contexts)
    return f"""{sys_inst}

# Requirements
- {' '.join(requirements)}

# Context
{ctx_text}

# Question
{question}

# Answer:
"""

def load_docx_files(data_dir: Path) -> List[Tuple[str, str]]:
    """
    data_dir 配下の .docx ファイルを読み込み、
    (ファイル名, テキスト全文) のリストを返す。
    """
    pairs = []
    data_dir.mkdir(exist_ok=True)
    for p in sorted(data_dir.glob("*.docx")):
        try:
            doc = Document(p)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            pairs.append((p.name, text))
        except Exception as e:
            print(f"❌ {p.name} の読み込みに失敗: {e}")
    return pairs

# ---------- エクスポート ----------
__all__ = [
    "normalize_ja_text",
    "split_text",
    "load_txt_files",
    "EmbeddingStore",
    "NumpyVectorDB",
    "ProcessedFilesSimple",
    "build_prompt",
    "load_docx_files",
]
