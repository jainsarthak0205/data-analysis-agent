"""CLI for data-analysis-agent."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from data_analysis_agent.agent import AgentConfig, DataAnalysisAgent
from data_analysis_agent.dataset import (
    Dataset,
    SUPPORTED_AGGS,
    format_dataframe,
    load_penguins,
)
from data_analysis_agent.llm import ClaudeClient, SUPPORTED_MODELS


def _build_dataset(args) -> Dataset:
    if args.csv:
        import pandas as pd
        return Dataset(pd.read_csv(args.csv))
    return load_penguins()


def _cmd_ask(args):
    dataset = _build_dataset(args)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY is not set. Export it before running the agent.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "(Tests use a deterministic fake client and do not need this.)",
            file=sys.stderr,
        )
        return 2
    llm = ClaudeClient(api_key=api_key, model=args.model)
    agent = DataAnalysisAgent(dataset=dataset, llm=llm, config=AgentConfig(max_steps=args.max_steps))
    result = agent.ask(args.question)
    if args.json:
        print(json.dumps({
            "answer": result.answer,
            "notes": result.notes,
            "n_tool_calls": result.n_tool_calls,
            "steps": [
                {
                    "step": s.step,
                    "tool_name": s.tool_name,
                    "tool_input": s.tool_input,
                    "tool_result_preview": s.tool_result[:240],
                    "is_error": s.is_error,
                    "final_text": s.final_text or None,
                }
                for s in result.steps
            ],
        }, indent=2))
        return 0
    if args.show_trace:
        for s in result.steps:
            if s.tool_name:
                print(f"[step {s.step}] {s.tool_name}({json.dumps(s.tool_input)})")
                for line in s.tool_result.splitlines()[:6]:
                    print(f"    {line}")
                extra = len(s.tool_result.splitlines()) - 6
                if extra > 0:
                    print(f"    ... ({extra} more lines)")
        if result.notes:
            print("\n[notes]")
            for i, n in enumerate(result.notes, 1):
                print(f"  {i}. {n}")
        print()
    print(result.answer)
    return 0


def _cmd_info(args):
    dataset = _build_dataset(args)
    cols = dataset.columns()
    print(f"dataset: {dataset.n_rows} rows x {dataset.n_cols} cols")
    for c in cols:
        print(f"  {c.name}: dtype={c.dtype}  null={c.n_null}  unique={c.n_unique}")
    return 0


def _cmd_describe(args):
    dataset = _build_dataset(args)
    desc = dataset.describe_column(args.column)
    if args.json:
        print(json.dumps(desc, indent=2))
        return 0
    print(f"column {desc['name']!r} ({desc['dtype']}):")
    for k, v in desc.items():
        if k in ("name", "dtype"):
            continue
        if k == "value_counts":
            print("  top values:")
            for cat, cnt in v.items():
                print(f"    {cat}: {cnt}")
        else:
            print(f"  {k}: {v}")
    return 0


def _cmd_query(args):
    dataset = _build_dataset(args)
    df = dataset.query(args.expr, max_rows=args.max_rows)
    if args.json:
        print(df.to_json(orient="records", indent=2))
    else:
        print(format_dataframe(df))
    return 0


def _cmd_aggregate(args):
    dataset = _build_dataset(args)
    df = dataset.aggregate(group_by=args.group_by.split(","), value_col=args.value_col, agg=args.agg)
    if args.json:
        print(df.to_json(orient="records", indent=2))
    else:
        print(format_dataframe(df))
    return 0


def _cmd_correlation(args):
    dataset = _build_dataset(args)
    result = dataset.correlation(args.col1, args.col2)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"pearson r({result['col1']}, {result['col2']}) = {result['r']} (n={result['n_used']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="data-analysis-agent",
        description="Agentic pandas analyst over the Palmer Penguins dataset (Claude tool use; defaults to Opus 4.8).",
    )
    p.add_argument("--csv", default=None, help="Path to a custom CSV (else the bundled Palmer Penguins)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ap = sub.add_parser("ask", help="Ask a natural-language question; the agent answers with pandas")
    ap.add_argument("question")
    ap.add_argument(
        "--model",
        default=ClaudeClient.DEFAULT_MODEL,
        help=(
            "Claude model id. Recommended: "
            + ", ".join(SUPPORTED_MODELS)
            + ". Any other Claude model id is also accepted."
        ),
    )
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--show-trace", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.set_defaults(func=_cmd_ask)

    ip = sub.add_parser("info", help="Show dataset schema")
    ip.set_defaults(func=_cmd_info)

    dp = sub.add_parser("describe", help="Per-column summary (no LLM)")
    dp.add_argument("column")
    dp.add_argument("--json", action="store_true")
    dp.set_defaults(func=_cmd_describe)

    qp = sub.add_parser("query", help="Filter rows with a pandas query expression (no LLM)")
    qp.add_argument("expr")
    qp.add_argument("--max-rows", type=int, default=10)
    qp.add_argument("--json", action="store_true")
    qp.set_defaults(func=_cmd_query)

    agp = sub.add_parser("aggregate", help="Groupby + named aggregation (no LLM)")
    agp.add_argument("group_by", help="Comma-separated column names")
    agp.add_argument("value_col")
    agp.add_argument("agg", choices=list(SUPPORTED_AGGS))
    agp.add_argument("--json", action="store_true")
    agp.set_defaults(func=_cmd_aggregate)

    cp = sub.add_parser("correlation", help="Pearson r between two numeric columns (no LLM)")
    cp.add_argument("col1")
    cp.add_argument("col2")
    cp.add_argument("--json", action="store_true")
    cp.set_defaults(func=_cmd_correlation)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
