"""Dense retrieval with a pluggable encoder.

The encoder is a protocol so the retriever can be tested without downloading a
model. The real implementation wraps sentence-transformers and is imported lazily,
because importing torch takes seconds and the BM25 path must stay fast.

Prefix conventions differ per model family and are not optional: e5 needs
``"query: "`` / ``"passage: "``, bge wants a retrieval instruction on the query,
gte needs neither. Getting this wrong silently costs points, so the prefixes are
part of the model spec and are asserted in tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from src.data import ARTIFACTS_DIR
from src.types import Chunk, RetrievedSpan, sort_retrieved


class Encoder(Protocol):
    """Turns text into L2-normalised vectors."""

    name: str
    dim: int

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ModelSpec:
    hf_name: str
    query_prefix: str = ""
    passage_prefix: str = ""


#: The three embedders chosen for a size spread (docs/05-retrieval.md).
MODEL_SPECS: dict[str, ModelSpec] = {
    "bge-base-en-v1.5": ModelSpec(
        hf_name="BAAI/bge-base-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "gte-base": ModelSpec(hf_name="thenlper/gte-base"),
    "e5-large-v2": ModelSpec(
        hf_name="intfloat/e5-large-v2",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
}


def prepare_passages(spec: ModelSpec, texts: Sequence[str]) -> list[str]:
    """Apply the model family's passage prefix."""
    return [spec.passage_prefix + text for text in texts]


def prepare_query(spec: ModelSpec, text: str) -> str:
    """Apply the model family's query prefix or instruction."""
    return spec.query_prefix + text


class SentenceTransformerEncoder:
    """Lazy sentence-transformers wrapper."""

    def __init__(
        self,
        model: str = "bge-base-en-v1.5",
        *,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        if model not in MODEL_SPECS:
            raise ValueError(f"unknown embedder {model!r}; available: {sorted(MODEL_SPECS)}")
        self.model = model
        self.spec = MODEL_SPECS[model]
        self.batch_size = batch_size
        self.device = device
        self.name = model
        self._model = None
        self.dim = 0

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.spec.hf_name, device=self.device)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        inputs = prepare_passages(self.spec, texts)
        vectors = model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        model = self._load()
        return np.asarray(
            model.encode(
                [prepare_query(self.spec, text)],
                batch_size=1,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0],
            dtype=np.float32,
        )


def build_encoder(model: str, **kwargs) -> Encoder:
    return SentenceTransformerEncoder(model, **kwargs)


def _ids_hash(chunk_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for chunk_id in chunk_ids:
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


@dataclass
class DenseRetriever:
    """Cosine-similarity retrieval over cached chunk embeddings."""

    model: str = "bge-base-en-v1.5"
    batch_size: int = 64
    name: str = "dense"
    cache_dir: Path | None = None
    encoder: Encoder | None = None

    _chunks: list[Chunk] = field(default_factory=list, init=False, repr=False)
    _matrix: np.ndarray | None = field(default=None, init=False, repr=False)
    cache_hit: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.encoder is None:
            self.encoder = build_encoder(self.model, batch_size=self.batch_size)

    @property
    def cache_path(self) -> Path:
        root = Path(self.cache_dir or ARTIFACTS_DIR / "embeddings")
        return root / f"{self.model}-{_ids_hash([c.chunk_id for c in self._chunks])}.npz"

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            return

        path = self.cache_path
        if path.exists():
            vectors = np.load(path)["vectors"]
            if vectors.shape[0] == len(self._chunks):
                self._matrix = vectors.astype(np.float32)
                self.cache_hit = True
                return

        assert self.encoder is not None
        vectors = self.encoder.encode_passages([chunk.text for chunk in self._chunks])
        self._matrix = np.asarray(vectors, dtype=np.float32)
        self.cache_hit = False
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self._matrix)

    def search(self, query: str, k: int = 10) -> list[RetrievedSpan]:
        if not self._chunks or self._matrix is None or k <= 0:
            return []
        assert self.encoder is not None
        query_vector = self.encoder.encode_query(query)
        scores = self._matrix @ query_vector

        order = np.argsort(-scores)[:k]
        hits = [
            RetrievedSpan(
                file_path=self._chunks[int(i)].file_path,
                start=self._chunks[int(i)].start,
                end=self._chunks[int(i)].end,
                score=float(scores[int(i)]),
                chunk_id=self._chunks[int(i)].chunk_id,
            )
            for i in order
            if scores[int(i)] > 0.0
        ]
        return sort_retrieved(hits)

    def __len__(self) -> int:
        return len(self._chunks)
