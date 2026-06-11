"""Library tests for data-analysis-agent.

All tests run offline. The agentic loop is driven by FakeLLMClient with
a scripted sequence of tool calls — no Anthropic API key required.

Ground-truth Palmer Penguins reference values (from the public
allisonhorst/palmerpenguins repo, used by countless tutorials):
  - 344 rows, 8 columns
  - 3 species: Adelie (152), Gentoo (124), Chinstrap (68)
  - 3 islands: Biscoe, Dream, Torgersen
  - Mean Gentoo body mass ~ 5076 g (heaviest)
  - r(bill_length_mm, body_mass_g) ~ +0.59
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from data_analysis_agent.agent import AgentConfig, DataAnalysisAgent, TOOL_DEFINITIONS
from data_analysis_agent.dataset import (
    Dataset,
    SUPPORTED_AGGS,
    bundled_penguins_path,
    format_dataframe,
    load_penguins,
)
from data_analysis_agent.llm import FakeLLMClient, FakeStep, ToolCall


PKG_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    return load_penguins()


# ---------- dataset ----------

def test_bundled_penguins_loads(dataset):
    assert dataset.n_rows == 344
    assert dataset.n_cols == 8


def test_expected_columns(dataset):
    names = [c.name for c in dataset.columns()]
    assert names == [
        "species", "island", "bill_length_mm", "bill_depth_mm",
        "flipper_length_mm", "body_mass_g", "sex", "year",
    ]


def test_species_counts_match_published(dataset):
    """Reference: Adelie 152, Gentoo 124, Chinstrap 68."""
    desc = dataset.describe_column("species")
    vc = desc["value_counts"]
    assert vc["Adelie"] == 152
    assert vc["Gentoo"] == 124
    assert vc["Chinstrap"] == 68


def test_islands_are_three(dataset):
    vc = dataset.describe_column("island")["value_counts"]
    assert set(vc.keys()) == {"Biscoe", "Dream", "Torgersen"}


def test_describe_numeric_has_quartiles(dataset):
    d = dataset.describe_column("bill_length_mm")
    for key in ("mean", "std", "min", "25%", "50%", "75%", "max"):
        assert key in d


def test_describe_unknown_column_raises(dataset):
    with pytest.raises(ValueError):
        dataset.describe_column("ghost")


def test_query_filters_rows(dataset):
    df = dataset.query("species == 'Gentoo'", max_rows=50)
    assert len(df) == 50  # capped
    assert df["species"].unique().tolist() == ["Gentoo"]


def test_query_default_caps_at_row_limit(dataset):
    """Whatever max_rows the LLM asks for, the dataset caps at row_cap (50)."""
    df = dataset.query("species != 'extinct'", max_rows=999)
    assert len(df) == 50  # row_cap


def test_query_bad_expression_raises(dataset):
    with pytest.raises(ValueError):
        dataset.query("syntactically !! bad")


def test_query_empty_expression_raises(dataset):
    with pytest.raises(ValueError):
        dataset.query("")


def test_aggregate_mean_body_mass_by_species(dataset):
    """Reference: Gentoo is heaviest at ~5076g; Adelie ~3700g; Chinstrap ~3733g."""
    df = dataset.aggregate(group_by=["species"], value_col="body_mass_g", agg="mean")
    # Result is sorted descending by the aggregated column.
    rows = df.to_dict(orient="records")
    assert rows[0]["species"] == "Gentoo"
    assert 5000 <= rows[0]["body_mass_g_mean"] <= 5200
    species_to_mean = {r["species"]: r["body_mass_g_mean"] for r in rows}
    assert 3600 <= species_to_mean["Adelie"] <= 3800
    assert 3650 <= species_to_mean["Chinstrap"] <= 3850


def test_aggregate_count_works_on_any_column(dataset):
    df = dataset.aggregate(group_by=["species"], value_col="island", agg="count")
    counts = {r["species"]: r["island_count"] for r in df.to_dict(orient="records")}
    assert counts["Adelie"] == 152


def test_aggregate_rejects_non_numeric_for_mean(dataset):
    with pytest.raises(ValueError):
        dataset.aggregate(group_by=["species"], value_col="island", agg="mean")


def test_aggregate_rejects_unknown_agg(dataset):
    with pytest.raises(ValueError):
        dataset.aggregate(group_by=["species"], value_col="body_mass_g", agg="cube_root")


def test_aggregate_rejects_empty_group_by(dataset):
    with pytest.raises(ValueError):
        dataset.aggregate(group_by=[], value_col="body_mass_g", agg="mean")


def test_aggregate_two_dimensional(dataset):
    df = dataset.aggregate(
        group_by=["species", "sex"], value_col="body_mass_g", agg="mean",
    )
    # 3 species x 2 sexes (+ possibly a NA row) — at least 6 grouping combinations
    assert len(df) >= 6


def test_correlation_bill_length_vs_body_mass(dataset):
    """Reference: Pearson r ~ 0.59 between bill_length_mm and body_mass_g."""
    result = dataset.correlation("bill_length_mm", "body_mass_g")
    assert result["method"] == "pearson"
    assert 0.55 <= result["r"] <= 0.65
    assert result["n_used"] > 300


def test_correlation_rejects_non_numeric(dataset):
    with pytest.raises(ValueError):
        dataset.correlation("species", "body_mass_g")


def test_correlation_rejects_unknown_column(dataset):
    with pytest.raises(ValueError):
        dataset.correlation("ghost", "body_mass_g")


def test_format_dataframe_renders_pipes(dataset):
    df = dataset.query("species == 'Adelie'", max_rows=2)
    text = format_dataframe(df)
    assert " | " in text
    # Header + separator + 2 rows
    assert len(text.splitlines()) == 4


def test_format_dataframe_empty():
    import pandas as pd
    assert format_dataframe(pd.DataFrame()) == "(empty)"


def test_dataset_rejects_empty():
    import pandas as pd
    with pytest.raises(ValueError):
        Dataset(pd.DataFrame())


def test_supported_aggs_are_canonical():
    assert SUPPORTED_AGGS == ("mean", "median", "sum", "min", "max", "count", "std", "nunique")


def test_bundled_path_exists():
    assert bundled_penguins_path().exists()


# ---------- tool definitions ----------

def test_tool_definitions_complete():
    names = {t["name"] for t in TOOL_DEFINITIONS}
    assert names == {
        "list_columns", "describe_column", "query",
        "aggregate", "correlation", "take_note",
    }
    for t in TOOL_DEFINITIONS:
        assert "description" in t and len(t["description"]) > 0
        assert t["input_schema"]["type"] == "object"


def test_aggregate_tool_schema_enumerates_agg_funcs():
    spec = next(t for t in TOOL_DEFINITIONS if t["name"] == "aggregate")
    assert set(spec["input_schema"]["properties"]["agg"]["enum"]) == set(SUPPORTED_AGGS)


# ---------- agent loop with FakeLLMClient ----------

def test_agent_one_shot_no_tools(dataset):
    fake = FakeLLMClient(script=[FakeStep(final_text="The answer is 42.")])
    agent = DataAnalysisAgent(dataset=dataset, llm=fake)
    result = agent.ask("dumb question")
    assert result.answer == "The answer is 42."
    assert result.n_tool_calls == 0


def test_agent_runs_aggregate_for_heaviest_species(dataset):
    """Canonical trace: aggregate -> take_note -> answer."""
    fake = FakeLLMClient(script=[
        FakeStep(tool_calls=[ToolCall(
            id="t1", name="aggregate",
            input={"group_by": ["species"], "value_col": "body_mass_g", "agg": "mean"},
        )]),
        FakeStep(tool_calls=[ToolCall(
            id="t2", name="take_note",
            input={"text": "Gentoo is the heaviest at ~5076g (mean body_mass_g)."},
        )]),
        FakeStep(final_text="Gentoo penguins are the heaviest with a mean body mass of about 5076 g."),
    ])
    agent = DataAnalysisAgent(dataset=dataset, llm=fake)
    result = agent.ask("Which species is heaviest?")
    assert "Gentoo" in result.answer
    assert result.n_tool_calls == 2
    agg = next(s for s in result.steps if s.tool_name == "aggregate")
    assert "Gentoo" in agg.tool_result
    assert "5076" in agg.tool_result or "5075" in agg.tool_result
    assert len(result.notes) == 1


def test_agent_runs_correlation(dataset):
    fake = FakeLLMClient(script=[
        FakeStep(tool_calls=[ToolCall(
            id="t1", name="correlation",
            input={"col1": "bill_length_mm", "col2": "body_mass_g"},
        )]),
        FakeStep(final_text="r = +0.59 — moderately positive correlation."),
    ])
    agent = DataAnalysisAgent(dataset=dataset, llm=fake)
    result = agent.ask("Is bill length related to body mass?")
    step = result.steps[0]
    assert step.tool_name == "correlation"
    assert "pearson" in step.tool_result.lower()
    # 0.59 should appear in the rendered output
    assert "0.5" in step.tool_result


def test_agent_handles_unknown_column_gracefully(dataset):
    fake = FakeLLMClient(script=[
        FakeStep(tool_calls=[ToolCall(
            id="t1", name="describe_column",
            input={"name": "ghost_column"},
        )]),
        FakeStep(final_text="There's no such column; here are the real ones."),
    ])
    agent = DataAnalysisAgent(dataset=dataset, llm=fake)
    result = agent.ask("describe ghost_column")
    step = result.steps[0]
    assert step.is_error
    assert "no column named" in step.tool_result


def test_agent_handles_query_typo_gracefully(dataset):
    fake = FakeLLMClient(script=[
        FakeStep(tool_calls=[ToolCall(id="t1", name="query", input={"expr": "@@@"})]),
        FakeStep(final_text="That query was malformed."),
    ])
    agent = DataAnalysisAgent(dataset=dataset, llm=fake)
    result = agent.ask("filter on garbage")
    assert result.steps[0].is_error
    assert "bad query" in result.steps[0].tool_result


def test_agent_take_note_populates_result_notes(dataset):
    fake = FakeLLMClient(script=[
        FakeStep(tool_calls=[ToolCall(id="t1", name="take_note", input={"text": "first."})]),
        FakeStep(tool_calls=[ToolCall(id="t2", name="take_note", input={"text": "second."})]),
        FakeStep(final_text="ok"),
    ])
    agent = DataAnalysisAgent(dataset=dataset, llm=fake)
    result = agent.ask("note things")
    assert result.notes == ["first.", "second."]


def test_agent_respects_max_steps(dataset):
    forever = lambda hist: FakeStep(  # noqa: E731
        tool_calls=[ToolCall(id=f"loop_{len(hist)}", name="list_columns", input={})]
    )
    fake = FakeLLMClient(script=forever)
    agent = DataAnalysisAgent(dataset=dataset, llm=fake, config=AgentConfig(max_steps=3))
    result = agent.ask("loop")
    assert result.stopped_at_max_steps
    assert result.n_tool_calls == 3


def test_agent_rejects_empty_question(dataset):
    fake = FakeLLMClient(script=[])
    agent = DataAnalysisAgent(dataset=dataset, llm=fake)
    with pytest.raises(ValueError):
        agent.ask("")


def test_agent_unknown_tool_errors_cleanly(dataset):
    fake = FakeLLMClient(script=[
        FakeStep(tool_calls=[ToolCall(id="t1", name="train_model", input={})]),
        FakeStep(final_text="no such tool"),
    ])
    agent = DataAnalysisAgent(dataset=dataset, llm=fake)
    result = agent.ask("can you train a model?")
    assert result.steps[0].is_error
    assert "unknown tool" in result.steps[0].tool_result


# ---------- CLI ----------

def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "data_analysis_agent", *args],
        cwd=PKG_ROOT,
        env={**os.environ, "PYTHONPATH": str(PKG_ROOT)},
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_info():
    r = _run_cli("info")
    assert r.returncode == 0, r.stderr
    assert "species" in r.stdout
    assert "body_mass_g" in r.stdout


def test_cli_aggregate_mean_body_mass():
    r = _run_cli("aggregate", "species", "body_mass_g", "mean", "--json")
    assert r.returncode == 0, r.stderr
    rows = json.loads(r.stdout)
    # Top row is Gentoo (heaviest)
    assert rows[0]["species"] == "Gentoo"


def test_cli_correlation_json():
    r = _run_cli("correlation", "bill_length_mm", "body_mass_g", "--json")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert 0.55 <= out["r"] <= 0.65


def test_cli_ask_without_api_key_errors_clearly():
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["PYTHONPATH"] = str(PKG_ROOT)
    r = subprocess.run(
        [sys.executable, "-m", "data_analysis_agent", "ask", "anything?"],
        cwd=PKG_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 2
    assert "ANTHROPIC_API_KEY" in r.stderr
