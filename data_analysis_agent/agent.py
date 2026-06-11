"""DataAnalysisAgent — answers numerical questions about a tabular dataset.

Tools (all read-only, all schema-validated):
- list_columns(): every column with dtype, null count, n_unique
- describe_column(name): per-column stats (numeric: percentiles; categorical: top values)
- query(expr, max_rows): df.query() with a row cap, no eval / no mutation
- aggregate(group_by, value_col, agg): groupby with named aggregations only
- correlation(col1, col2): Pearson r between two numeric columns
- take_note(text): scratchpad, surfaced in result.notes

The agent runs a manual agentic loop on top of a pluggable LLM client
— same architecture as the prior agentic projects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data_analysis_agent.dataset import (
    Dataset,
    SUPPORTED_AGGS,
    format_dataframe,
)
from data_analysis_agent.llm import LLMClient, ToolCall


SYSTEM_PROMPT = """You are a careful data analyst working with a tabular dataset.

You have read-only access to the data through these tools:

- `list_columns()` — every column with dtype, null count, and uniqueness
- `describe_column(name)` — numeric: mean/std/min/quartiles/max; categorical: top value counts
- `query(expr, max_rows=10)` — pandas df.query() with a row cap; expressions like `species == "Adelie" and body_mass_g > 4000`
- `aggregate(group_by, value_col, agg)` — groupby + named aggregation. `agg` must be one of: mean, median, sum, min, max, count, std, nunique
- `correlation(col1, col2)` — Pearson r between two numeric columns
- `take_note(text)` — private scratchpad; surfaced in the final result

Method:
1. Call `list_columns()` once early if you don't already know the schema.
2. For a "what's the average X by Y?" question, use `aggregate`, NOT a Python loop.
3. For a "are X and Y related?" question, use `correlation`.
4. For "show me some rows where ...", use `query`.
5. Use `take_note` after each numerical finding so your reasoning is visible in the final result.
6. Answer with concrete numbers from the data, not approximations from your memory. Cite the tool result that produced each number.

Be terse. Skip preamble like "I'll analyze..." or "Let me check...". Start with the answer."""


# ---------- tool schemas ----------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "list_columns",
        "description": "List every column with its dtype, null count, and uniqueness. Call this first if you don't know the schema.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "describe_column",
        "description": (
            "Return per-column summary statistics. Numeric columns: mean, std, min, 25%, 50%, 75%, max. "
            "Categorical columns: top 20 value counts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query",
        "description": (
            "Run a pandas `df.query(expr)` filter and return up to `max_rows` rows. "
            "Examples: `species == \"Adelie\"`, `body_mass_g > 5000 and sex == \"male\"`, "
            "`island == \"Biscoe\" and bill_length_mm < 40`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expr": {"type": "string", "description": "pandas query() expression"},
                "max_rows": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["expr"],
            "additionalProperties": False,
        },
    },
    {
        "name": "aggregate",
        "description": (
            f"Group by one or more columns and aggregate a value column. "
            f"`agg` must be one of: {', '.join(SUPPORTED_AGGS)}. "
            f"For numeric aggs (mean/median/sum/etc) the value_col must be numeric; "
            f"`count` works on any column."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "One or more column names to group by",
                },
                "value_col": {"type": "string"},
                "agg": {"type": "string", "enum": list(SUPPORTED_AGGS)},
            },
            "required": ["group_by", "value_col", "agg"],
            "additionalProperties": False,
        },
    },
    {
        "name": "correlation",
        "description": "Compute Pearson correlation between two numeric columns. Returns r and the sample size used.",
        "input_schema": {
            "type": "object",
            "properties": {
                "col1": {"type": "string"},
                "col2": {"type": "string"},
            },
            "required": ["col1", "col2"],
            "additionalProperties": False,
        },
    },
    {
        "name": "take_note",
        "description": "Write a short note to your scratchpad. Use one note per concrete finding so you can cite them in the final answer.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "minLength": 1}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class StepLog:
    step: int
    tool_name: Optional[str] = None
    tool_input: Dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""
    is_error: bool = False
    final_text: str = ""


@dataclass
class AgentResult:
    answer: str
    notes: List[str] = field(default_factory=list)
    steps: List[StepLog] = field(default_factory=list)
    stopped_at_max_steps: bool = False

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.tool_name is not None)


@dataclass
class AgentConfig:
    max_steps: int = 10
    max_tokens_per_turn: int = 4096


class DataAnalysisAgent:
    def __init__(
        self,
        dataset: Dataset,
        llm: LLMClient,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        self.dataset = dataset
        self.llm = llm
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    # ---- tool dispatch ----------------------------------------------

    def _dispatch_tool(
        self,
        name: str,
        args: Dict[str, Any],
        notes: List[str],
    ) -> tuple[str, bool]:
        try:
            if name == "list_columns":
                cols = self.dataset.columns()
                lines = [f"dataset: {self.dataset.n_rows} rows x {self.dataset.n_cols} cols"]
                for c in cols:
                    lines.append(
                        f"  {c.name}: dtype={c.dtype}  null={c.n_null}  unique={c.n_unique}"
                    )
                return "\n".join(lines), False

            if name == "describe_column":
                col = args.get("name", "")
                desc = self.dataset.describe_column(col)
                lines = [f"column {desc['name']!r} ({desc['dtype']}):"]
                for k, v in desc.items():
                    if k in ("name", "dtype"):
                        continue
                    if k == "value_counts":
                        lines.append("  top values:")
                        for cat, cnt in v.items():
                            lines.append(f"    {cat}: {cnt}")
                    else:
                        lines.append(f"  {k}: {v}")
                return "\n".join(lines), False

            if name == "query":
                expr = args.get("expr", "")
                max_rows = args.get("max_rows", 10)
                df = self.dataset.query(expr, max_rows=max_rows)
                n_matched = len(df)
                header = f"matched (showing up to {max_rows}, returned {n_matched}):"
                return f"{header}\n{format_dataframe(df)}", False

            if name == "aggregate":
                group_by = args.get("group_by", [])
                if isinstance(group_by, str):
                    group_by = [group_by]
                value_col = args.get("value_col", "")
                agg = args.get("agg", "")
                df = self.dataset.aggregate(group_by=group_by, value_col=value_col, agg=agg)
                return format_dataframe(df), False

            if name == "correlation":
                result = self.dataset.correlation(
                    col1=args.get("col1", ""), col2=args.get("col2", "")
                )
                return (
                    f"pearson r({result['col1']}, {result['col2']}) "
                    f"= {result['r']} (n={result['n_used']})"
                ), False

            if name == "take_note":
                text = str(args.get("text", "")).strip()
                if not text:
                    return "error: empty note", True
                notes.append(text)
                return f"saved (now {len(notes)} note(s))", False

            return f"error: unknown tool {name!r}", True

        except ValueError as e:
            return f"error: {e}", True
        except Exception as e:
            return f"error: {type(e).__name__}: {e}", True

    # ---- main loop --------------------------------------------------

    def ask(self, question: str) -> AgentResult:
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": question.strip()},
        ]
        steps: List[StepLog] = []
        notes: List[str] = []

        for step_idx in range(1, self.config.max_steps + 1):
            response = self.llm.complete(
                system=self.system_prompt,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                max_tokens=self.config.max_tokens_per_turn,
            )

            if not response.tool_calls:
                final = response.text or "(no answer)"
                steps.append(StepLog(step=step_idx, final_text=final))
                return AgentResult(answer=final, notes=notes, steps=steps)

            assistant_blocks: List[Dict[str, Any]] = []
            if response.text:
                assistant_blocks.append({"type": "text", "text": response.text})
            for call in response.tool_calls:
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.input,
                })
            messages.append({"role": "assistant", "content": assistant_blocks})

            tool_results: List[Dict[str, Any]] = []
            for call in response.tool_calls:
                rendered, is_error = self._dispatch_tool(call.name, call.input, notes)
                steps.append(StepLog(
                    step=step_idx,
                    tool_name=call.name,
                    tool_input=dict(call.input),
                    tool_result=rendered,
                    is_error=is_error,
                ))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": rendered,
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": tool_results})

        return AgentResult(
            answer="(stopped: max_steps reached without final answer)",
            notes=notes,
            steps=steps,
            stopped_at_max_steps=True,
        )
