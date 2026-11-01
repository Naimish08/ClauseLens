"""Streamlit app: a thin wrapper over the same config-driven pipeline.

    make app

Nothing about retrieval lives here; the app only chooses a config, indexes once
and renders results (or an abstention) with source document and offsets.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import load_config
from src.data import REPO_ROOT, load_mini_cuad
from src.pipeline import Pipeline

CONFIG_DIR = REPO_ROOT / "configs"


def available_configs() -> list[Path]:
    paths = sorted(CONFIG_DIR.glob("*.yaml")) + sorted((CONFIG_DIR / "generated").glob("*.yaml"))
    return [path for path in paths if path.name.startswith(("baseline", "final", "iter_"))]


@st.cache_resource(show_spinner="Building index…")
def load_pipeline(config_path: str) -> tuple[Pipeline, dict[str, str]]:
    config = load_config(Path(config_path))
    _, documents = load_mini_cuad(mode=config.data.read_mode)
    pipeline = Pipeline.build(config)
    pipeline.index(documents.values())
    contents = {path: document.content for path, document in documents.items()}
    return pipeline, contents


def main() -> None:
    st.set_page_config(page_title="ClauseLens", page_icon="📄", layout="wide")
    st.title("ClauseLens")
    st.caption("Clause-level retrieval over LegalBench-RAG (CUAD subset), with character offsets.")

    configs = available_configs()
    if not configs:
        st.error("No configs found under configs/. Run `make setup && make data` first.")
        return

    with st.sidebar:
        config_path = st.selectbox(
            "Pipeline config",
            options=configs,
            format_func=lambda path: path.name,
        )
        top_k = st.slider("Passages to return", min_value=1, max_value=20, value=10)
        st.markdown(
            "Every stage is chosen by the config file. Rebuild the index by changing "
            "the config and reloading."
        )

    query = st.text_input(
        "Question",
        placeholder="What is the governing law of this contract?",
    )

    if not query:
        st.info("Ask a question about the indexed contracts.")
        return

    pipeline, contents = load_pipeline(str(config_path))
    result = pipeline.search(query, k=top_k)

    if result.abstained:
        st.warning(f"No passage is relevant enough — {result.reason}")
        return

    st.caption(
        f"{len(result.spans)} passages · {result.timings_ms.get('retrieval', 0):.1f} ms retrieval"
    )
    for rank, span in enumerate(result.spans, start=1):
        text = contents.get(span.file_path, "")[span.start : span.end]
        with st.container(border=True):
            header, score = st.columns([0.8, 0.2])
            header.markdown(f"**{rank}. {Path(span.file_path).name}**")
            score.markdown(f"`score {span.score:.3f}`")
            st.caption(f"`{span.file_path}` · characters {span.start}–{span.end}")
            st.write(text if len(text) <= 1200 else text[:1200] + "…")


if __name__ == "__main__":
    main()
