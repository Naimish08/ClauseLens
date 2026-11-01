#!/usr/bin/env python3
"""Mine candidates for the unanswerable set.

Prints a clause-presence matrix over the 29 indexed contracts, then proposes:

* type 1 candidates — clause probes that match nothing anywhere
* type 3 candidates — (document, clause) pairs where the clause exists in several
  contracts but not in this one
* type 4 candidates — near-miss pairs where a related clause exists in the same
  document but the asked-for one does not

The output is a starting point for authoring, not the answer: every candidate must
be read in context before it becomes an entry in ``data/unanswerable.jsonl``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_mini_cuad  # noqa: E402

PROBES: dict[str, tuple[str, ...]] = {
    "governing_law": (r"governing law", r"governed by the laws"),
    "termination_for_convenience": (
        r"terminate (?:this agreement )?(?:at any time )?(?:for convenience|without cause)",
        r"termination for convenience",
    ),
    "termination_for_cause": (
        r"(?:terminate|termination) (?:this agreement )?for cause",
        r"termination for breach",
    ),
    "liability_cap": (r"limitation of liability", r"aggregate liability", r"shall not exceed"),
    "indemnification": (r"indemnif", r"hold harmless"),
    "non_compete": (r"non-compet", r"shall not compete", r"covenant not to compete"),
    "non_solicitation": (r"non-solicit", r"solicit"),
    "confidentiality": (r"confidential",),
    "anti_assignment": (r"may not assign", r"shall not assign", r"assignment of this agreement"),
    "audit_rights": (r"\baudit\b", r"books and records"),
    "license_grant": (
        r"hereby grants?",
        r"grants? to .{0,30}a (?:non-exclusive|exclusive)?\s*licen[cs]e",
    ),
    "most_favored_nation": (r"most favored", r"most favoured"),
    "right_of_first_refusal": (r"right of first refus", r"right of first offer"),
    "exclusivity": (r"\bexclusiv",),
    "renewal": (r"\brenew",),
    "force_majeure": (r"force majeure", r"act of god"),
    "warranty": (r"\bwarrant(?:y|ies)\b", r"represents and warrants"),
    "insurance": (r"\binsurance\b",),
    "escrow": (r"\bescrow\b",),
    "arbitration": (r"\barbitrat",),
    "liquidated_damages": (r"liquidated damages",),
    "survival": (r"\bsurviv",),
    "expense_reimbursement": (r"reimburs",),
    "royalty": (r"royalt",),
    "minimum_commitment": (r"minimum (?:purchase|order|commitment|annual)",),
    "data_protection": (r"data protection", r"personal data", r"gdpr"),
    "ip_ownership": (r"work made for hire", r"work for hire", r"owns? all right, title"),
    "severability": (r"severab",),
    "entire_agreement": (r"entire agreement",),
    "taxes": (r"\btaxes\b|\btax\b",),
    "subcontracting": (r"subcontract",),
}

NEAR_MISS_PAIRS: tuple[tuple[str, str], ...] = (
    ("termination_for_cause", "termination_for_convenience"),
    ("confidentiality", "data_protection"),
    ("non_compete", "non_solicitation"),
    ("indemnification", "insurance"),
    ("license_grant", "royalty"),
    ("minimum_commitment", "royalty"),
    ("audit_rights", "insurance"),
    ("renewal", "termination_for_convenience"),
    ("right_of_first_refusal", "exclusivity"),
    ("ip_ownership", "escrow"),
)


def presence(documents: dict) -> dict[str, dict[str, bool]]:
    matrix: dict[str, dict[str, bool]] = {}
    for file_path, document in documents.items():
        matrix[file_path] = {
            clause: any(re.search(pattern, document.content, re.I) for pattern in patterns)
            for clause, patterns in PROBES.items()
        }
    return matrix


def short(file_path: str) -> str:
    return file_path.split("/")[-1][:58]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-others",
        type=int,
        default=5,
        help="minimum other docs for a type-3 candidate",
    )
    parser.add_argument("--evidence", type=int, default=90, help="characters of context to print")
    args = parser.parse_args()

    _, documents = load_mini_cuad()
    matrix = presence(documents)

    counts = {clause: sum(row[clause] for row in matrix.values()) for clause in PROBES}
    print(f"{len(documents)} documents\n")
    print("clause presence:")
    for clause, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {count:2d}/{len(documents)}  {clause}")

    print("\n=== type 1 candidates: absent everywhere ===")
    for clause, count in counts.items():
        if count == 0:
            print(f"  {clause}: {PROBES[clause]}")

    print(f"\n=== type 3 candidates (clause in >= {args.min_others} other docs, absent here) ===")
    for file_path, row in sorted(matrix.items()):
        missing = [
            (clause, counts[clause])
            for clause, present in row.items()
            if not present and counts[clause] >= args.min_others
        ]
        missing.sort(key=lambda item: -item[1])
        if missing:
            print(f"  {short(file_path)}")
            print("     missing: " + ", ".join(f"{c}({n})" for c, n in missing[:8]))

    print("\n=== type 4 candidates (near-miss pairs) ===")
    for file_path, row in sorted(matrix.items()):
        for present_clause, absent_clause in NEAR_MISS_PAIRS:
            if row.get(present_clause) and not row.get(absent_clause):
                print(f"  {short(file_path)}: has {present_clause}, lacks {absent_clause}")

    print("\n=== evidence for absent-everywhere probes ===")
    for clause, count in counts.items():
        if count != 0:
            continue
        print(f"\n  [{clause}] no match anywhere; sample corpus phrases:")
        text = next(iter(documents.values())).content
        for pattern in PROBES[clause]:
            match = re.search(pattern, text, re.I)
            if match:
                start = max(0, match.start() - 40)
                print(f"    {text[start:match.end() + 40]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
