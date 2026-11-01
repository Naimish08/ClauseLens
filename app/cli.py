"""ClauseLens CLI: ask one question against a config-driven index.

    python -m app.cli --config configs/baseline.yaml --query "What is the governing law?"

Prints ranked passages with source document, character offsets and an excerpt, or
an explicit abstention message.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.data import load_mini_cuad  # noqa: E402
from src.pipeline import Pipeline, PipelineResult  # noqa: E402


def excerpt(text: str, width: int = 240) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= width else collapsed[:width] + "…"


def format_result(result: PipelineResult, contents: dict[str, str], width: int = 240) -> str:
    if result.abstained:
        return (
            "No passage is relevant enough.\n"
            f"reason: {result.reason}\n"
            f"query:  {result.query}"
        )

    lines = [f"query: {result.query}", ""]
    for rank, span in enumerate(result.spans, start=1):
        text = contents.get(span.file_path, "")[span.start : span.end]
        lines.append(
            f"{rank:2d}. {span.file_path}  [{span.start}:{span.end}]  score={span.score:.4f}"
        )
        lines.append(f"    {excerpt(text, width)}")
    return "\n".join(lines)


def build_pipeline(config_path: Path) -> tuple[Pipeline, dict[str, str]]:
    config = load_config(config_path)
    _, documents = load_mini_cuad(mode=config.data.read_mode)
    pipeline = Pipeline.build(config)
    pipeline.index(documents.values())
    contents = {path: document.content for path, document in documents.items()}
    return pipeline, contents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None, help="override the config's top_k")
    parser.add_argument("--width", type=int, default=240, help="excerpt width in characters")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    pipeline, contents = build_pipeline(args.config)
    result = pipeline.search(args.query, k=args.top_k)

    if args.json:
        payload = result.as_dict()
        for item in payload["spans"]:  # type: ignore[index]
            text = contents.get(item["file_path"], "")[item["start"] : item["end"]]
            item["excerpt"] = excerpt(text, args.width)
        print(json.dumps(payload, indent=2))
    else:
        print(format_result(result, contents, width=args.width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
