"""FastAPI service for data-analysis-agent."""
from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data_analysis_agent.agent import AgentConfig, AgentResult, DataAnalysisAgent
from data_analysis_agent.dataset import (
    Dataset,
    SUPPORTED_AGGS,
    format_dataframe,
    load_penguins,
)
from data_analysis_agent.llm import ClaudeClient, LLMClient


DEFAULT_MODEL = os.environ.get("DATA_AGENT_MODEL", ClaudeClient.DEFAULT_MODEL)
MAX_STEPS = int(os.environ.get("DATA_AGENT_MAX_STEPS", "10"))


class _State:
    def __init__(self):
        self.dataset: Optional[Dataset] = None
        self.llm: Optional[LLMClient] = None

    def get_dataset(self) -> Dataset:
        if self.dataset is None:
            self.dataset = load_penguins()
        return self.dataset

    def get_llm(self) -> LLMClient:
        if self.llm is not None:
            return self.llm
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="ANTHROPIC_API_KEY is not set. Configure it on the server.",
            )
        self.llm = ClaudeClient(api_key=api_key, model=DEFAULT_MODEL)
        return self.llm


state = _State()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    max_steps: int = Field(MAX_STEPS, ge=1, le=20)
    show_trace: bool = False


class StepView(BaseModel):
    step: int
    tool_name: Optional[str] = None
    tool_input: dict = Field(default_factory=dict)
    tool_result_preview: str = ""
    is_error: bool = False
    final_text: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    notes: List[str] = Field(default_factory=list)
    n_tool_calls: int
    stopped_at_max_steps: bool
    steps: List[StepView] = Field(default_factory=list)


class ColumnView(BaseModel):
    name: str
    dtype: str
    n_null: int
    n_unique: int
    summary: dict


class InfoResponse(BaseModel):
    n_rows: int
    n_cols: int
    columns: List[ColumnView]
    model: str
    api_key_configured: bool


class DescribeResponse(BaseModel):
    column: dict


class QueryRequest(BaseModel):
    expr: str = Field(..., min_length=1, max_length=2000)
    max_rows: int = Field(10, ge=1, le=50)


class QueryResponse(BaseModel):
    expr: str
    n_returned: int
    rows: List[dict]


class AggregateRequest(BaseModel):
    group_by: List[str] = Field(..., min_length=1)
    value_col: str
    agg: str = Field(..., description="One of: " + ", ".join(SUPPORTED_AGGS))


class AggregateResponse(BaseModel):
    group_by: List[str]
    value_col: str
    agg: str
    rows: List[dict]


class CorrelationRequest(BaseModel):
    col1: str
    col2: str


class CorrelationResponse(BaseModel):
    col1: str
    col2: str
    method: str
    n_used: int
    r: float


app = FastAPI(
    title="data-analysis-agent",
    description="Agentic pandas analyst over the Palmer Penguins dataset (Claude Opus 4.8).",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
def info():
    ds = state.get_dataset()
    cols = ds.columns()
    return InfoResponse(
        n_rows=ds.n_rows,
        n_cols=ds.n_cols,
        columns=[ColumnView(**c.__dict__) for c in cols],
        model=DEFAULT_MODEL,
        api_key_configured=bool(os.environ.get("ANTHROPIC_API_KEY")),
    )


@app.get("/columns/{name}", response_model=DescribeResponse)
def describe(name: str):
    ds = state.get_dataset()
    try:
        return DescribeResponse(column=ds.describe_column(name))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    ds = state.get_dataset()
    try:
        df = ds.query(req.expr, max_rows=req.max_rows)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return QueryResponse(
        expr=req.expr,
        n_returned=len(df),
        rows=df.to_dict(orient="records"),
    )


@app.post("/aggregate", response_model=AggregateResponse)
def aggregate(req: AggregateRequest):
    ds = state.get_dataset()
    try:
        df = ds.aggregate(group_by=req.group_by, value_col=req.value_col, agg=req.agg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AggregateResponse(
        group_by=req.group_by,
        value_col=req.value_col,
        agg=req.agg,
        rows=df.to_dict(orient="records"),
    )


@app.post("/correlation", response_model=CorrelationResponse)
def correlation(req: CorrelationRequest):
    ds = state.get_dataset()
    try:
        return CorrelationResponse(**ds.correlation(req.col1, req.col2))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    ds = state.get_dataset()
    llm = state.get_llm()
    agent = DataAnalysisAgent(dataset=ds, llm=llm, config=AgentConfig(max_steps=req.max_steps))
    result: AgentResult = agent.ask(req.question)
    return AskResponse(
        answer=result.answer,
        notes=result.notes,
        n_tool_calls=result.n_tool_calls,
        stopped_at_max_steps=result.stopped_at_max_steps,
        steps=[
            StepView(
                step=s.step,
                tool_name=s.tool_name,
                tool_input=s.tool_input,
                tool_result_preview=s.tool_result[:300] if req.show_trace else "",
                is_error=s.is_error,
                final_text=s.final_text or None,
            )
            for s in result.steps
        ] if req.show_trace else [],
    )
