"""Offline end-to-end demo. Uses FakeLLMClient — no API key needed."""
from __future__ import annotations

from data_analysis_agent import (
    AgentConfig,
    DataAnalysisAgent,
    FakeLLMClient,
    load_penguins,
)
from data_analysis_agent.llm import FakeStep, ToolCall


def main():
    dataset = load_penguins()
    print(f"dataset: {dataset.n_rows} rows x {dataset.n_cols} cols")
    cols = dataset.columns()
    for c in cols:
        print(f"  {c.name}: {c.dtype} (nulls={c.n_null}, unique={c.n_unique})")
    print()

    question = (
        "Which penguin species is heaviest on average, and is bill length "
        "correlated with body mass?"
    )
    print(f"Q: {question}\n")

    fake = FakeLLMClient(script=[
        FakeStep(tool_calls=[ToolCall(id="t1", name="list_columns", input={})]),
        FakeStep(tool_calls=[ToolCall(
            id="t2", name="aggregate",
            input={"group_by": ["species"], "value_col": "body_mass_g", "agg": "mean"},
        )]),
        FakeStep(tool_calls=[ToolCall(
            id="t3", name="correlation",
            input={"col1": "bill_length_mm", "col2": "body_mass_g"},
        )]),
        FakeStep(tool_calls=[ToolCall(
            id="t4", name="take_note",
            input={"text": "Gentoo heaviest: 5076g mean body mass; r(bill_length, body_mass) ~ 0.59."},
        )]),
        FakeStep(final_text=(
            "Gentoo penguins are the heaviest on average at ~5076 g (Chinstrap "
            "~3733 g, Adelie ~3701 g). Bill length and body mass are moderately "
            "positively correlated (Pearson r ~ 0.59, n=342) — longer-billed "
            "penguins do tend to be heavier."
        )),
    ])

    agent = DataAnalysisAgent(dataset=dataset, llm=fake, config=AgentConfig(max_steps=8))
    result = agent.ask(question)

    print("--- trace ---")
    for s in result.steps:
        if s.tool_name:
            print(f"[step {s.step}] {s.tool_name}({s.tool_input})")
            for line in s.tool_result.splitlines()[:6]:
                print(f"    {line}")
            extra = len(s.tool_result.splitlines()) - 6
            if extra > 0:
                print(f"    ... ({extra} more lines)")
        else:
            print(f"[step {s.step}] final answer.")

    print("\n[notes]")
    for i, n in enumerate(result.notes, 1):
        print(f"  {i}. {n}")

    print(f"\n--- answer ---\n{result.answer}\n")
    print(f"({result.n_tool_calls} tool calls, "
          f"max_steps_reached={result.stopped_at_max_steps})")


if __name__ == "__main__":
    main()
