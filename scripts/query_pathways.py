#!/usr/bin/env python3
"""Thin CLI wrapper for single-compound AraCyc pathway queries."""

from __future__ import annotations

import argparse
from pathlib import Path

from pathway_query_shared import load_preprocessed_state, print_result, run_query


def repl(state: dict, top_k: int, no_fuzzy: bool) -> None:
    """Interactive REPL for pathway queries."""

    print("\nAraCyc-first pathway query REPL. Type a compound name or 'quit' to exit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"quit", "exit", "q"}:
            break
        result = run_query(query, state, top_k=top_k, no_fuzzy=no_fuzzy)
        print_result(result, top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query AraCyc-first pathway mappings.")
    parser.add_argument("--workdir", default=".", help="Workspace root directory.")
    parser.add_argument("--query", "-q", default="", help="Single query (non-interactive).")
    parser.add_argument("--top-k", type=int, default=10, help="Max pathways to show.")
    parser.add_argument("--no-fuzzy", action="store_true", help="Disable typo correction for plain-text queries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    state = load_preprocessed_state(workdir)

    if args.query:
        result = run_query(args.query, state, top_k=args.top_k, no_fuzzy=args.no_fuzzy)
        print_result(result, args.top_k)
    else:
        repl(state, args.top_k, args.no_fuzzy)


if __name__ == "__main__":
    main()
