FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir uv==0.5.*

COPY requirements.txt /app/requirements.txt
RUN uv pip install --system -r /app/requirements.txt

COPY data_analysis_agent /app/data_analysis_agent
COPY pyproject.toml README.md /app/

EXPOSE 8000

# Pass ANTHROPIC_API_KEY at runtime:
#   docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY -p 8000:8000 data-analysis-agent
CMD ["uvicorn", "data_analysis_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
