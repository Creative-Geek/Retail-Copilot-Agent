# Retail Analytics Copilot

A local, privacy-first AI agent for retail analytics using LangGraph, DSPy, and SQLite.

## Overview

This agent answers questions about the Northwind database and local policy documents. It uses a hybrid approach:

- **RAG**: For policy and calendar questions (BM25 retrieval).
- **SQL**: For data queries (SQLite).
- **Hybrid**: For questions requiring both (e.g., "revenue during Summer 2017").

### Cache Busting Note

To prevent Ollama from returning cached responses during testing, all queries are prefixed with a unique `[Query ID: xxxxxxxx]` identifier. This ensures fresh LLM responses on each run.

## Architecture

The agent is built with **LangGraph** and consists of 7 nodes:

```text
┌──────────┐
│  Router  │ ─────────────────────────────────┐
└────┬─────┘                                  │
     │ (rag/hybrid)                           │ (sql)
     ▼                                        │
┌──────────┐                                  │
│ Retriever│                                  │
└────┬─────┘                                  │
     │                                        │
     ▼                                        │
┌──────────┐                                  │
│ Planner  │ ─────┐                           │
└────┬─────┘      │ (rag)                     │
     │ (hybrid)   │                           │
     ▼            │                           ▼
┌──────────┐      │                    ┌──────────┐
│ SQL Gen  │ ◄────┴────────────────────│  SQL Gen │
└────┬─────┘                           └────┬─────┘
     │                                      │
     ▼                                      ▼
┌──────────┐                           ┌──────────┐
│ SQL Exec │ ─────┐ (error)            │ SQL Exec │
└────┬─────┘      │                    └────┬─────┘
     │ (success)  ▼                         │
     │      ┌──────────┐                    │
     │      │  Repair  │ (retry)            │
     │      └──────────┘                    │
     ▼            │                         ▼
┌──────────┐      │                    ┌──────────┐
│Synthesize│ ◄────┘                    │Synthesize│
└──────────┘                           └──────────┘
```

### Node Details

1. **Router**: Rule-based classification of questions as RAG, SQL, or Hybrid based on keywords.
2. **Retriever**: BM25-based document retrieval with section-level chunking from `docs/`.
3. **Planner**: Extracts date constraints and KPI formulas from retrieved documents.
4. **SQL Gen**: Generates SQLite queries using templates for known patterns + LLM fallback (DSPy optimized).
5. **SQL Exec**: Executes the query against the Northwind database.
6. **Repair**: Retries SQL generation on error (max 2 retries) with error context.
7. **Synthesizer**: Formats the final answer with proper type conversion and citations.

## DSPy Optimization

I optimized the **GenerateSQL** module using `dspy.BootstrapFewShot`.

### Metrics

| Metric         | Before | After | Improvement |
| -------------- | ------ | ----- | ----------- |
| Valid SQL Rate | 40%    | 80%   | +40%        |

### Before Optimization

- The model frequently generated invalid SQL syntax
- Incorrect table names and missing quotes around "Order Details"
- Poor date filtering and join conditions

### After Optimization

- Valid SQL execution rate improved significantly
- Correct table joins (`Orders` + `"Order Details"` + `Products`)
- Proper date formatting with BETWEEN clauses

### Training Examples

The optimizer uses 6 training examples covering:

- Revenue calculations with date ranges
- Top N queries with aggregations
- Category-based filtering
- Gross margin calculations

## Tracing & Debugging

The agent writes a trace log to `agent_trace.jsonl` for each run:

```json
{
  "session_id": "20251126_151647",
  "step": 1,
  "node": "router",
  "duration_ms": 0.01,
  "input": { "question": "..." },
  "output": { "mode": "hybrid" }
}
```

This enables replay and debugging of agent executions.

## Setup & Usage

UV is an extremely fast Python package and project manager, written in Rust.

It's used exactly like web technologies (like npm/pnpm/yarn), but for Python, I used `uv` to manage dependencies and run scripts.

1. **Install dependencies**:

   ```bash
   uv sync
   ```

2. **Start the Ollama server** (in a separate terminal):

   ```bash
   ollama serve
   ```

   Then pull the model (first time only):

   ```bash
   ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
   ```

3. **Run the agent**:

   ```bash
   uv run python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
   ```

4. **Run extended tests** (includes evaluation, extended, and edge case tests):

   ```bash
   uv run python test_agent.py
   ```

   Test options:

   - `--quick` or `--eval`: Run only the 6 evaluation questions
   - `--extended`: Run extended test cases
   - `--edge`: Run edge case tests
   - (no args): Run all tests

5. **Run DSPy optimization** (optional):

   ```bash
   uv run python -m agent.optimize_sql
   ```

6. **Interactive Query GUI** (optional):

   ```bash
   uv run python query.py
   ```

   This opens a simple GUI where you can type questions and get answers from the agent interactively.

## Files Structure

```text
├── agent/
│   ├── graph_hybrid.py       # Main LangGraph workflow
│   ├── dspy_signatures.py    # DSPy signature definitions
│   ├── optimize_sql.py       # DSPy optimization script
│   ├── compiled_sql_gen.json # Compiled SQL module
│   ├── rag/
│   │   └── retrieval.py      # BM25 document retrieval
│   └── tools/
│       └── sqlite_tool.py    # SQLite database access
├── data/
│   └── northwind.sqlite      # Northwind SQLite database (not included in this repo)
├── docs/
│   ├── catalog.md            # Product catalog info
│   ├── kpi_definitions.md    # KPI formulas
│   ├── marketing_calendar.md # Campaign dates
│   └── product_policy.md     # Return policies
├── pyproject.toml            # uv project configuration
├── uv.lock                   # uv lock file (dependency versions)
├── run_agent_hybrid.py       # CLI entry point
├── query.py                  # Interactive GUI for querying the agent
├── test_agent.py             # Test suite with eval/extended/edge cases
├── sample_questions_hybrid_eval.jsonl  # Evaluation questions (not included in this repo)
└── agent_trace.jsonl         # Execution trace log (not included in this repo)
```

## Trade-offs

- **Rule-based Router**: I use deterministic keyword matching for reliability and speed. DSPy optimization is focused on SQL generation where it has the most impact.
- **Template-based SQL**: For reliability, common question patterns use SQL templates instead of LLM generation. This improves accuracy but limits flexibility.
- **Retrieval**: Simple BM25 with section-level chunking is used. A semantic vector store would improve retrieval for complex queries.
- **Cost Approximation**: I use `CostOfGoods ≈ 0.7 * UnitPrice` for gross margin calculations as specified in, since the Northwind database doesn't include cost data.
