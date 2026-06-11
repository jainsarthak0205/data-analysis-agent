# data-analysis-agent

> **An agentic pandas analyst. Claude inspects the schema, runs queries, groups and aggregates, and computes correlations against the bundled Palmer Penguins dataset — answering natural-language questions with concrete numbers from the data. Defaults to Claude Opus 4.8 but supports any Claude model (Opus 4.6/4.7/4.8, Sonnet 4.6, Haiku 4.5, Fable 5).**

Different shape from [sql-agent](https://github.com/jainsarthak0205/sql-agent): instead of writing SQL strings, the agent calls **structured pandas operations** through six typed tools. The tool surface is the safety boundary — no `eval()` on raw Python, no DataFrame mutation, no arbitrary callables in aggregation.

Same pluggable-LLM + manual-loop + scripted-fake-for-tests pattern as the prior agentic projects ([sql-agent](https://github.com/jainsarthak0205/sql-agent), [research-agent](https://github.com/jainsarthak0205/research-agent), [code-review-agent](https://github.com/jainsarthak0205/code-review-agent), [debate-agent](https://github.com/jainsarthak0205/debate-agent)).

## The six tools

| Tool | What it does | Safety boundary |
| --- | --- | --- |
| `list_columns()` | Every column with dtype, null count, n_unique. Call once early. | — |
| `describe_column(name)` | Numeric: mean / std / min / 25% / 50% / 75% / max. Categorical: top 20 value counts. | Column name validated against the schema. |
| `query(expr, max_rows)` | `df.query(expr)` — pandas filter expression. | `expr` runs through pandas' parser (no Python `eval`); result capped at 50 rows. |
| `aggregate(group_by, value_col, agg)` | Group by one or more columns, aggregate one value column. | `agg` must be one of `mean / median / sum / min / max / count / std / nunique` — **no `agg(lambda)`**. Numeric aggs require a numeric value column. |
| `correlation(col1, col2)` | Pearson r between two numeric columns + n_used. | Both columns required to be numeric. |
| `take_note(text)` | Scratchpad, surfaced in `result.notes`. | — |

## Install

```bash
# With uv (recommended — much faster)
uv venv
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt    # for tests

# Or with plain pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Dataset

The bundled `penguins.csv` is the **Palmer Penguins** dataset (Allison Horst, 2020), sourced from [allisonhorst/palmerpenguins](https://github.com/allisonhorst/palmerpenguins) — the modern replacement for the over-used Iris dataset.

| Statistic | Value |
| --- | --- |
| Rows | 344 |
| Columns | 8 (species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex, year) |
| Species | Adelie (152), Gentoo (124), Chinstrap (68) |
| Islands | Biscoe, Dream, Torgersen |
| Null values | 2-11 per measurement column (real-world missing-data structure) |

**Known reference statistics** (so tests can assert exact numbers and a portfolio reviewer can sanity-check the agent's answers):

| Question | Answer |
| --- | --- |
| Which species is heaviest on average? | **Gentoo** (~5076 g) |
| How does that compare to others? | Chinstrap ~3733 g, Adelie ~3701 g |
| Are bill length and body mass correlated? | Yes — Pearson **r ≈ 0.595** (n=342) |
| Which species lives on Torgersen island? | Only Adelie |

## CLI

The static subcommands work without an API key (they share the `Dataset` operations with the agent):

```bash
python -m data_analysis_agent info
# dataset: 344 rows x 8 cols
#   species: dtype=str  null=0  unique=3
#   island: dtype=str  null=0  unique=3
#   bill_length_mm: dtype=float64  null=2  unique=164
#   ...

python -m data_analysis_agent describe species
# column 'species' (str):
#   n_null: 0
#   n_unique: 3
#   top values:
#     Adelie: 152
#     Gentoo: 124
#     Chinstrap: 68

python -m data_analysis_agent aggregate species body_mass_g mean --json
# [{"species":"Gentoo","body_mass_g_mean":5076.016}, ...]

python -m data_analysis_agent correlation bill_length_mm body_mass_g
# pearson r(bill_length_mm, body_mass_g) = 0.5951 (n=342)

# The agent itself needs an API key
export ANTHROPIC_API_KEY=sk-ant-...
python -m data_analysis_agent ask "Which species is heaviest, and what predicts body mass best?" --show-trace
```

To use your own CSV, point the CLI at it:

```bash
python -m data_analysis_agent --csv my_data.csv info
python -m data_analysis_agent --csv my_data.csv ask "..."
```

## Python API

```python
import os
import pandas as pd
from data_analysis_agent import (
    AgentConfig,
    ClaudeClient,
    DataAnalysisAgent,
    Dataset,
    load_penguins,
)

dataset = load_penguins()  # or: Dataset(pd.read_csv("my.csv"))
llm = ClaudeClient(api_key=os.environ["ANTHROPIC_API_KEY"])
agent = DataAnalysisAgent(dataset=dataset, llm=llm, config=AgentConfig(max_steps=10))

result = agent.ask("Which species is heaviest, and how confident can we be?")
print(result.answer)
for note in result.notes:
    print(" -", note)
```

### Offline tests with `FakeLLMClient`

```python
from data_analysis_agent import FakeLLMClient, DataAnalysisAgent, load_penguins
from data_analysis_agent.llm import FakeStep, ToolCall

fake = FakeLLMClient(script=[
    FakeStep(tool_calls=[ToolCall(
        id="t1", name="aggregate",
        input={"group_by": ["species"], "value_col": "body_mass_g", "agg": "mean"},
    )]),
    FakeStep(final_text="Gentoo penguins are the heaviest at ~5076 g."),
])
agent = DataAnalysisAgent(dataset=load_penguins(), llm=fake)
print(agent.ask("which species is heaviest?").answer)
```

## HTTP API

```bash
ANTHROPIC_API_KEY=sk-ant-... uvicorn data_analysis_agent.api:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | What it does |
| --- | --- | --- |
| `/health` | GET | Liveness probe |
| `/info` | GET | Schema, model, `api_key_configured` |
| `/columns/{name}` | GET | One column's stats (no LLM) |
| `/query` | POST | Filter rows with a pandas expression (no LLM) |
| `/aggregate` | POST | Groupby + named aggregation (no LLM) |
| `/correlation` | POST | Pearson r between two numeric columns (no LLM) |
| `/ask` | POST | Run the agent on a natural-language question |

```bash
curl -s -X POST http://localhost:8000/aggregate \
  -H 'content-type: application/json' \
  -d '{"group_by": ["species"], "value_col": "body_mass_g", "agg": "mean"}'

curl -s -X POST http://localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "Which species is heaviest?", "show_trace": true}'
```

`/ask` returns **503** if `ANTHROPIC_API_KEY` isn't set. The static endpoints work without any key — they share the `Dataset` operations with the agent (same code path).

Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | (none) | Required for `/ask` and CLI `ask` |
| `DATA_AGENT_MODEL` | `claude-opus-4-8` | Override the Claude model ID. Recommended: `claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5`, `claude-fable-5`. Any other Claude model id is also accepted. |
| `DATA_AGENT_MAX_STEPS` | `10` | Max agentic-loop iterations per question |

## Deploy

```bash
docker build -t data-analysis-agent .
docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY data-analysis-agent

curl -X POST http://localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "Are flipper length and body mass correlated?"}'
```

## Design notes

- **The tool surface is the safety boundary.** A general-purpose "run pandas code" tool would let the LLM mutate the DataFrame, run arbitrary computations, or quietly OOM by reading 100k rows. Six structured tools constrain the agent to operations the dataset can validate: column names are checked against the schema, `query` runs through pandas' parser (not `eval`), `aggregate` only accepts a named set of operations, results are capped at 50 rows.
- **Aggregations are sorted descending by the result column.** "Which species is heaviest?" should not require the agent to read the rows in arbitrary order — sorting by the aggregated column means the answer is in row 0. Small but meaningful: it shaves a tool call.
- **Correlations report `n_used` alongside `r`.** Real-world data has nulls; an `r` computed on 342 rows is more trustworthy than an `r` computed on 12. Surfacing `n_used` lets the agent caveat its conclusions appropriately.
- **The static endpoints (`/aggregate`, `/correlation`, `/query`) share code with the agent's tools.** Same `Dataset.aggregate(...)` is called whether the request comes in via the LLM's `tool_use` block or via `POST /aggregate`. This means the HTTP API can be used as a deterministic, no-cost fallback when you don't want to spend tokens.
- **Adaptive thinking is on by default**, but auto-disabled on models that don't support it (e.g. Haiku 4.5). On Opus 4.6+/Sonnet 4.6/Fable 5, `ClaudeClient` sets `thinking={"type": "adaptive"}`. See `SUPPORTED_MODELS` / `THINKING_MODELS` in `data_analysis_agent/llm.py`.
- **`data_analysis_agent/dataset.py`, not `data.py`.** The package has a `data_analysis_agent/data/` directory holding the CSV; naming the loader `data.py` would make it inaccessible (subpackage wins on import).

## Tests

```bash
pytest
```

All 52 tests run **offline** with no Anthropic API key — the agentic loop is exercised through `FakeLLMClient` with scripted tool-call sequences.

- **Library — dataset (24)**: Palmer Penguins loads with 344 rows × 8 columns and the expected column names; species counts match the published reference (Adelie 152, Gentoo 124, Chinstrap 68); islands are exactly {Biscoe, Dream, Torgersen}; `describe_column` returns quartiles for numerics; unknown column raises `ValueError`; `query` filters correctly, caps at `row_cap`, rejects bad expressions and empty strings; `aggregate(mean, body_mass_g)` returns Gentoo at the top with ~5076g and Adelie/Chinstrap in the ~3700g range; `count` works on any column; non-numeric value_col with `mean` rejected; unknown agg rejected; empty `group_by` rejected; 2-D groupby works; `correlation(bill_length_mm, body_mass_g)` returns r ≈ 0.59; correlation rejects non-numeric and unknown columns; `format_dataframe` renders pipes; empty DataFrame returns `(empty)`; empty Dataset rejected; `SUPPORTED_AGGS` is the canonical tuple; bundled path exists.
- **Library — tool definitions (2)**: every tool has a well-formed JSON schema; the `aggregate` tool's `agg` enum matches `SUPPORTED_AGGS` exactly so the LLM can only request supported operations.
- **Library — agent (9)**: agent returns text immediately when no tool calls are needed; canonical trace (`aggregate` → `take_note` → answer) for the heaviest-species question; `correlation` tool flow with r≈0.59 in the rendered output; unknown column errors propagate as `is_error=True` and the agent continues; bad query expressions propagate cleanly; multiple `take_note` calls populate `result.notes`; `max_steps` enforced on a loop; empty question rejected; unknown tool name returns an error.
- **Library — CLI (4)**: `info`, `aggregate species body_mass_g mean --json`, and `correlation` work without an API key and produce the expected values; `ask` without an API key exits 2 with a clear stderr.
- **API (13)**: `/health`, `/info`, `/columns/{species}` returns the right value counts; unknown column → 404; `/query` filters and bad expression returns 400; `/aggregate` returns Gentoo at the top with the right magnitude; invalid `agg` → 400; `/correlation` returns r ≈ 0.59; non-numeric correlation → 400; `/ask` with an injected fake LLM runs end-to-end; `/ask` without an API key returns **503**; `/ask` validates `question` min length.

## License

MIT — see [LICENSE](LICENSE).
