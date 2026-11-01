"""The pipeline: the only component that knows about every stage.

It receives documents (never queries-with-spans) at index time and a plain string
at query time. That signature is the enforcement mechanism for invariant I4 —
ground truth cannot leak into inference because it is not reachable.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from src.abstain import Abstainer, QueryFeatures, build_abstainer
from src.chunking import Chunker, build_chunker, validate_chunks
from src.config import PipelineConfig
from src.data import Document, derive_category
from src.retrieval import Reranker, Retriever, build_reranker, build_retriever
from src.types import Chunk, RetrievedSpan


@dataclass(frozen=True, slots=True)
class IndexStats:
    documents: int
    chunks: int
    characters: int

    def as_dict(self) -> dict[str, int]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "characters": self.characters,
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """What a caller gets back: ranked spans, or an explained abstention."""

    query: str
    spans: tuple[RetrievedSpan, ...]
    abstained: bool
    reason: str
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def top(self) -> RetrievedSpan | None:
        return self.spans[0] if self.spans else None

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "abstained": self.abstained,
            "reason": self.reason,
            "spans": [
                {
                    "file_path": span.file_path,
                    "start": span.start,
                    "end": span.end,
                    "score": span.score,
                    "chunk_id": span.chunk_id,
                }
                for span in self.spans
            ],
            "timings_ms": self.timings_ms,
        }


class Pipeline:
    """Config-driven search over a fixed document set."""

    def __init__(
        self,
        config: PipelineConfig,
        chunker: Chunker,
        retriever: Retriever,
        abstainer: Abstainer,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        self.chunker = chunker
        self.retriever = retriever
        self.abstainer = abstainer
        self.reranker = reranker
        self._chunk_ids: set[str] = set()
        self._chunks: tuple[Chunk, ...] = ()
        self._indexed = False
        self.index_stats = IndexStats(0, 0, 0)

    @classmethod
    def build(cls, config: PipelineConfig) -> Pipeline:
        return cls(
            config=config,
            chunker=build_chunker(config.chunking.name, config.chunking.params),
            retriever=build_retriever(config.retrieval.name, config.retrieval.params),
            abstainer=build_abstainer(config.abstain.name, config.abstain.params),
            reranker=build_reranker(config.rerank.name, config.rerank.params),
        )

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def index(self, documents: Iterable[Document]) -> IndexStats:
        chunks: list[Chunk] = []
        characters = 0
        n_documents = 0

        for document in documents:
            n_documents += 1
            characters += len(document.content)
            document_chunks = self.chunker.chunk(document)
            validate_chunks(document, document_chunks)
            chunks.extend(document_chunks)

        self.retriever.index(chunks)
        self._chunk_ids = {chunk.chunk_id for chunk in chunks}
        self._chunks = tuple(chunks)
        self._indexed = True
        self.index_stats = IndexStats(n_documents, len(chunks), characters)
        return self.index_stats

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """The indexed chunks, for retrieval-free diagnostics (span containment)."""
        return self._chunks

    def indexed_chunk_ids(self) -> set[str]:
        """Copy of the indexed chunk IDs; used to check invariant I2."""
        return set(self._chunk_ids)

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #
    def prepare_query(self, query: str) -> str:
        """Apply query-time text handling from the config.

        ``strip_doc_hint`` removes the ``"Consider the ...;"`` prefix. It is off by
        default because that prefix is benchmark metadata (ADR-010); enabling it is
        an ablation, not a default.
        """
        return derive_category(query) if self.config.query.strip_doc_hint else query

    def search(self, query: str, k: int | None = None) -> PipelineResult:
        """Search, returning at most ``k`` spans (default: config ``top_k``).

        Evaluation passes its largest cut-off so a single ranking can be scored at
        several k; serving uses the config default.
        """
        if not self._indexed:
            raise RuntimeError("Pipeline.search() called before index()")

        prepared = self.prepare_query(query)
        depth = k if k is not None else self.config.top_k

        started = time.perf_counter()
        candidates: Sequence[RetrievedSpan] = self.retriever.search(prepared, k=depth)
        retrieval_ms = (time.perf_counter() - started) * 1000.0

        rerank_ms = 0.0
        if self.reranker is not None and candidates:
            started = time.perf_counter()
            candidates = self.reranker.rerank(prepared, candidates, k=depth)
            rerank_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        decision = self.abstainer.decide(
            QueryFeatures.from_retrieval(prepared, candidates)
        )
        abstain_ms = (time.perf_counter() - started) * 1000.0

        spans = () if decision.abstain else tuple(candidates[:depth])

        if spans:
            unknown = [span.chunk_id for span in spans if span.chunk_id not in self._chunk_ids]
            if unknown:
                raise RuntimeError(
                    f"retriever returned spans that were never indexed: {unknown[:3]}"
                )

        return PipelineResult(
            query=query,
            spans=spans,
            abstained=decision.abstain,
            reason=decision.reason,
            timings_ms={
                "retrieval": retrieval_ms,
                "rerank": rerank_ms,
                "abstain": abstain_ms,
            },
        )
