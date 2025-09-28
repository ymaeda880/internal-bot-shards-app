# lib/rag_utils.py
# ============================================
# このモジュールは RAG 用ユーティリティを提供します。
#
# 提供機能:
#  - split_text(text, chunk_size, overlap)
#  - EmbeddingStore(backend="openai"/"local",
#       model=None, *, openai_model=None, local_model=None).embed([...]) -> np.ndarray[float32]
#      * L2 正規化済み、次元は埋め込み結果から動的に決定（ハードコード禁止）
#      * ★ 後方互換: 旧API（openai_model/local_model）も受け付ける
#  - NumpyVectorDB(base_dir).add(vectors, metas) / search(query_vec, top_k, return_="similarity")
#      * 類似度は「大きいほど良い」に統一（cosine 既定、ベクトルは正規化想定）
#      * vectors.npy を安全に読み込み（壊れていれば自動隔離 .bad-YYYYMMDD-HHMMSS）
#  - ProcessedFilesSimple … 取り込み済ファイルの簡易トラッカー
#  - build_prompt(...): “資料外は分かりません” を強制する厳格プロンプト
#
# 重要修正:
#  1) 壊れた vectors.npy を自動退避（allow_pickle 誘発の例外を抑止）
#  2) 空入力時は (0,0) を返し、固定次元を廃止（次元不一致の芽を排除）
#  3) 回答は取得コンテキスト限定。根拠が無い事項は必ず「この資料からは分かりません」
#  4) ★ 後方互換: EmbeddingStore.__init__ に openai_model / local_model を追加
# ============================================

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable, Optional
from datetime import datetime, timezone

import json
import unicodedata
import numpy as np

# ========== 内部ユーティリティ ==========
def _warn(msg: str) -> None:
    """streamlit があれば警告、無ければ print。"""
    try:
        import streamlit as st  # type: ignore
        st.warning(msg)
    except Exception:
        print(f"[WARN] {msg}")

# ========== テキスト分割 ==========
def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[Tuple[str, int, int]]:
    """
    単純スライディング窓。返り値: [(chunk_text, start_idx, end_idx), ...]
    """
    text = unicodedata.normalize("NFKC", text or "")
    n = len(text)
    if n == 0 or chunk_size <= 0:
        return []

    step = max(1, chunk_size - max(0, overlap))
    out: List[Tuple[str, int, int]] = []
    i = 0
    while i < n:
        j = min(n, i + chunk_size)
        out.append((text[i:j], i, j))
        if j == n:
            break
        i += step
    return out

# ========== 埋め込み ==========
class EmbeddingStore:
    """
    backend:
      - "openai": OpenAI Embeddings API（既定: text-embedding-3-large / 3072 次元）
      - "local" : sentence-transformers（既定: intfloat/multilingual-e5-large）

    後方互換:
      旧コードの `EmbeddingStore(openai_model=..., local_model=...)` も動作します。
    """
    def __init__(
        self,
        backend: str = "openai",
        model: Optional[str] = None,
        *,
        openai_model: Optional[str] = None,   # ★ 旧パラメータ互換
        local_model: Optional[str] = None,    # ★ 旧パラメータ互換
    ):
        self.backend = backend
        # モデル決定ロジック（優先順位: model 明示 > 互換引数 > 既定）
        if model:
            chosen = model
        elif backend == "openai":
            chosen = openai_model or "text-embedding-3-large"
        else:
            chosen = local_model or "intfloat/multilingual-e5-large"
        self.model = chosen

        if self.backend not in ("openai", "local"):
            raise ValueError(f"Unknown embedding backend: {backend}")

        self._client = None
        self._st_model = None

    @staticmethod
    def _l2norm(X: np.ndarray) -> np.ndarray:
        X = X.astype("float32", copy=False)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return X / norms

    def embed(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        if len(texts) == 0:
            # 次元を固定しないため (0, 0) を返す
            return np.zeros((0, 0), dtype="float32")

        if self.backend == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:
                raise RuntimeError("openai SDK が見つかりません。backend='local' を使用してください。") from e

            if self._client is None:
                self._client = OpenAI()

            out: List[np.ndarray] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = self._client.embeddings.create(model=self.model, input=batch)
                vecs = np.array([d.embedding for d in resp.data], dtype="float32")
                out.append(vecs)
            embs = np.vstack(out)
            return self._l2norm(embs)

        # local backend
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError("sentence-transformers が見つかりません。backend='openai' を使用してください。") from e

        if self._st_model is None:
            self._st_model = SentenceTransformer(self.model)

        vecs = self._st_model.encode(texts, convert_to_numpy=True, batch_size=batch_size, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")

# ========== ベクトルDB ==========
class NumpyVectorDB:
    """
    vectors.npy: shape (N, d) float32（正規化済を推奨）
    meta.jsonl : 1 行 1 JSON
    """
    def __init__(self, base_dir: Path, metric: str = "cosine"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.base_dir / "vectors.npy"
        self.meta_path = self.base_dir / "meta.jsonl"
        self.metric = metric  # "cosine" | "dot" | "euclidean"

    # ---- 既存ベクトルの安全読み込み（壊れていれば自動隔離） ----
    def _load_existing_vectors(self) -> Optional[np.ndarray]:
        if not self.vec_path.exists():
            return None
        try:
            arr = np.load(self.vec_path, mmap_mode="r", allow_pickle=False)
        except Exception as e:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            bad = self.vec_path.with_suffix(f".npy.bad-{ts}")
            try:
                self.vec_path.rename(bad)
            except Exception:
                pass
            _warn(
                "vectors.npy の読み込みに失敗したため隔離しました: "
                f"{bad.name}\n原因: {e}\n→ シャードを再ベクトル化してください（大規模なら分割も検討）。"
            )
            return None

        if arr.ndim != 2 or arr.dtype != np.float32:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            bad = self.vec_path.with_suffix(f".npy.bad-{ts}")
            try:
                self.vec_path.rename(bad)
            except Exception:
                pass
            _warn(
                "vectors.npy の形式が不正（期待: 2D float32）。ファイルを隔離しました: "
                f"{bad.name}\n→ シャード再作成/再ベクトル化を実施してください。"
            )
            return None

        return arr

    # ---- メタ追記 ----
    def _append_meta(self, items: List[Dict[str, Any]]) -> None:
        if not items:
            return
        with self.meta_path.open("a", encoding="utf-8") as f:
            for m in items:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ---- 追加 ----
    def add(self, embeddings: np.ndarray, meta_items: List[Dict[str, Any]]) -> None:
        if embeddings is None or embeddings.size == 0:
            return
        embeddings = embeddings.astype("float32", copy=False)

        old = self._load_existing_vectors()
        if old is not None:
            if embeddings.shape[1] != old.shape[1]:
                raise ValueError(
                    f"次元不一致: new {embeddings.shape[1]} vs old {old.shape[1]} "
                    f"(別モデルの混在。シャードを分けるか作り直してください)"
                )
            new = np.vstack([np.asarray(old), embeddings])
        else:
            new = embeddings

        np.save(self.vec_path, new)
        self._append_meta(meta_items)

    # ---- 検索 ----
    def search(self, q: np.ndarray, top_k: int = 6, return_: str = "similarity"):
        """
        q: shape (1, d) または (d,)
        return_: "similarity" 固定推奨（大きいほど良いスコア）
        返り値: List[(row_idx, score, meta)]
        """
        if not self.vec_path.exists() or not self.meta_path.exists():
            return []

        X = self._load_existing_vectors()
        if X is None:
            return []

        q = np.asarray(q, dtype="float32").reshape(-1)
        if q.shape[0] != X.shape[1]:
            raise ValueError(f"クエリ次元がストアと不一致: q={q.shape[0]} vs d={X.shape[1]}")

        if self.metric in ("cosine", "dot"):
            score = (X @ q)
        else:
            dif = X - q
            score = -np.sqrt(np.sum(dif * dif, axis=1))

        k = int(max(1, min(top_k, X.shape[0])))
        idxs = np.argpartition(score, -k)[-k:]
        idxs = idxs[np.argsort(score[idxs])[::-1]]

        metas: List[Dict[str, Any]] = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        for i in idxs:
            try:
                metas.append(json.loads(lines[int(i)].rstrip("\n")))
            except Exception:
                metas.append({})

        return [(int(i), float(score[int(i)]), m) for i, m in zip(idxs.tolist(), metas)]

# ========== 取り込み済ファイル管理 ==========
class ProcessedFilesSimple:
    """
    processed_files.json で「取り込み済みファイル名」を管理する軽量トラッカー。
    フォーマット: {"done": ["a.pdf", "b.pdf", ...]}
    """
    def __init__(self, json_path: Path):
        self.path = Path(json_path)
        self._data = {"done": []}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and isinstance(obj.get("done"), list):
                self._data = {"done": [str(x) for x in obj["done"]]}
        except Exception as e:
            _warn(f"processed_files.json の読み込みに失敗: {e}")

    def is_done(self, filename: str) -> bool:
        return str(filename) in self._data["done"]

    def mark_done(self, filename: str) -> None:
        s = str(filename)
        if s not in self._data["done"]:
            self._data["done"].append(s)
            self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
