"""HTTP API tests for data-analysis-agent."""
from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    sys.modules.pop("data_analysis_agent.api", None)
    from data_analysis_agent import api  # noqa: WPS433
    return TestClient(api.app), api


def _inject_fake_llm(api_mod, steps):
    from data_analysis_agent.llm import FakeLLMClient
    api_mod.state.llm = FakeLLMClient(script=steps)


def test_health(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok"}


def test_info(client):
    c, _ = client
    body = c.get("/info").json()
    assert body["n_rows"] == 344
    assert body["n_cols"] == 8
    assert body["model"] == "claude-opus-4-8"


def test_describe_species(client):
    c, _ = client
    body = c.get("/columns/species").json()
    vc = body["column"]["value_counts"]
    assert vc["Adelie"] == 152


def test_describe_unknown_column_returns_404(client):
    c, _ = client
    assert c.get("/columns/ghost").status_code == 404


def test_query(client):
    c, _ = client
    body = c.post("/query", json={"expr": "species == 'Gentoo'", "max_rows": 5}).json()
    assert body["n_returned"] == 5
    for row in body["rows"]:
        assert row["species"] == "Gentoo"


def test_query_bad_expression_returns_400(client):
    c, _ = client
    r = c.post("/query", json={"expr": "@@@"})
    assert r.status_code == 400


def test_aggregate_returns_gentoo_heaviest(client):
    c, _ = client
    body = c.post("/aggregate", json={
        "group_by": ["species"],
        "value_col": "body_mass_g",
        "agg": "mean",
    }).json()
    # Sorted descending — Gentoo is first
    assert body["rows"][0]["species"] == "Gentoo"
    assert 5000 <= body["rows"][0]["body_mass_g_mean"] <= 5200


def test_aggregate_invalid_agg_returns_400(client):
    c, _ = client
    r = c.post("/aggregate", json={
        "group_by": ["species"],
        "value_col": "body_mass_g",
        "agg": "bogus",
    })
    assert r.status_code == 400


def test_correlation_endpoint(client):
    c, _ = client
    body = c.post("/correlation", json={"col1": "bill_length_mm", "col2": "body_mass_g"}).json()
    assert 0.55 <= body["r"] <= 0.65
    assert body["method"] == "pearson"


def test_correlation_non_numeric_returns_400(client):
    c, _ = client
    r = c.post("/correlation", json={"col1": "species", "col2": "body_mass_g"})
    assert r.status_code == 400


def test_ask_with_fake_llm(client):
    c, api = client
    from data_analysis_agent.llm import FakeStep, ToolCall
    _inject_fake_llm(api, [
        FakeStep(tool_calls=[ToolCall(
            id="t1", name="aggregate",
            input={"group_by": ["species"], "value_col": "body_mass_g", "agg": "mean"},
        )]),
        FakeStep(final_text="Gentoo penguins are the heaviest at ~5076 g."),
    ])
    r = c.post("/ask", json={"question": "which species is heaviest?", "show_trace": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "Gentoo" in body["answer"]
    assert body["n_tool_calls"] == 1


def test_ask_without_api_key_returns_503(client, monkeypatch):
    c, api = client
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    api.state.llm = None
    r = c.post("/ask", json={"question": "hi"})
    assert r.status_code == 503


def test_ask_validates_question_length(client):
    c, _ = client
    assert c.post("/ask", json={"question": ""}).status_code == 422
