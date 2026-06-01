# Thinking Agent — Decentralized MCP Architecture

A production-grade, modular AI agent system built with FastMCP and LangChain. The system splits a monolithic Stage 1 agent into two independent processes communicating over `streamable-http` via the Model Context Protocol (MCP).

The domain is agricultural advisory — the agent answers farming questions by retrieving grounded knowledge through a two-level hierarchical index, evaluating chunk relevance using genuine LLM-based Tree-of-Thought scoring, and improving its answers through a structured critique-correction reflection loop powered by MCP Sampling.

---

## Project Structure

```
thinking-agent/
├── pyproject.toml              ← uv workspace root
├── uv.lock                     ← auto-generated dependency lock file
├── README.md
├── REFLECTION.md
├── mcp_server/
│   ├── pyproject.toml          ← server dependencies
│   ├── main.py                 ← FastMCP server entry point
│   ├── knowledge_base.py       ← agricultural domain knowledge store
│   └── .env                    ← TAVILY_API_KEY
└── agent_client/
    ├── pyproject.toml          ← client dependencies
    ├── main.py                 ← LangChain agent + FastMCP client
    ├── agent_system.log        ← auto-generated persistent log file
    └── .env                    ← OPENROUTER_API_KEY
```

---

## System Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────┐
│                Agent Client                  │
│  LangChain Agent (create_agent)              │
│  FastMCP Client                              │
│    ├── sampling_handler  (LLM inference)     │
│    └── log_handler       (server log relay)  │
│  Local LLM (GPT-4o-mini via OpenRouter)      │
│  Logger → agent_system.log                  │
└─────────────────┬────────────────────────────┘
                  │  streamable-http (MCP protocol)
                  ▼
┌──────────────────────────────────────────────┐
│        MCP Server (FastMCP 3.3.1)            │
│                                              │
│  @resource knowledge://agriculture/docs/{q} │
│    ├── 1. Multi-query expansion (5 vectors)  │
│    ├── 2. Level-1 search: top-k doc ranking  │
│    ├── 3. Level-2 search: Jaccard similarity │
│    ├── 4. ToT evaluation via ctx.sample()    │
│    │       ├── Path 1: Analytical            │
│    │       ├── Path 2: Contextual            │
│    │       └── Path 3: Practical             │
│    └── 5. Tavily fallback (if ToT rejects)   │
│                                              │
│  @tool reflect_on_answer                     │
│    ├── Critic loop  → ctx.sample() → client  │
│    │     └── Validated: CritiqueResponse     │
│    └── Correction loop → ctx.sample() → client│
│          └── Validated: CorrectionResponse   │
└──────────────────────────────────────────────┘
```

---

## How It Works

### CRAG Resource Pipeline

When the agent queries the knowledge resource, the server runs a 5-stage pipeline:

1. **Multi-Query Expansion** — the raw query is expanded into 5 semantic search vectors covering different angles (best practices, challenges, productivity impact, modern techniques).

2. **Level-1 Hierarchical Search (top-k document ranking)** — each document's summary is scored by keyword overlap against all expanded queries. Only the top `k` documents (default: 3) are promoted, keeping chunk search manageable regardless of corpus size.

3. **Level-2 Hierarchical Search (chunk ranking)** — for each candidate document, every chunk is scored by Jaccard similarity against all expanded queries. Only chunks with similarity > 0 are returned, ranked descending.

4. **Tree-of-Thought Evaluation (LLM-based)** — for each retrieved chunk, `ctx.sample()` fires a real LLM call back to the client. The LLM reasons across 3 independent thought paths and returns a structured `ToTScoreResponse`:
   - **Path 1 — Analytical**: Does this chunk directly answer the query?
   - **Path 2 — Contextual**: Is the parent document topically relevant?
   - **Path 3 — Practical**: Is this chunk actionable and useful for a farmer?

   Each path is scored 0–10. Chunks with `avg >= 5` are retained. The response is validated against a Pydantic model before use.

5. **Tavily Fallback** — if no chunks survive the ToT filter, a live Tavily web search is triggered automatically to augment the response.

---

### Reflection Tool Pipeline

After the agent drafts an answer, it calls `reflect_on_answer` on the server. The server itself holds no LLM and makes no direct API calls — it delegates all inference back to the client via **MCP Sampling**:

1. **Critic loop** — server calls `ctx.sample()` with the original query and draft answer. The client LLM is instructed to return a structured JSON critique. The server validates the response against `CritiqueResponse` (Pydantic).

2. **Correction loop** — server calls `ctx.sample()` again, this time including the validated critique. The client LLM produces a corrected final answer validated against `CorrectionResponse` (Pydantic).

3. Both validated outputs are returned to the agent as a single structured response.

---

### Single-Client Architecture & Dual-Stream Logging

A single `fastmcp.Client` instance handles the entire session:
- `sampling_handler` — executes LLM inference locally when the server calls `ctx.sample()`
- `log_handler` — captures server `logging/message` notifications and writes them to `agent_system.log`

Every action is logged with timestamps and source tags:
```
[2026-06-01 16:53:05] [CLIENT] [INFO] Querying CRAG resource...
[2026-06-01 16:53:05] [SERVER] [INFO] Level-1 search: 3 candidate documents (top_k=3, corpus=5)
[2026-06-01 16:53:05] [SERVER] [INFO] ToT [doc_1] analytical=8 contextual=7 practical=6 avg=7.0
```

Both `[CLIENT]` and `[SERVER]` streams are written to `agent_client/agent_system.log`.

---

## Prerequisites

- Python 3.11+
- `uv` installed → https://astral.sh/uv
- OpenRouter API key → https://openrouter.ai
- Tavily API key → https://tavily.com

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/daviddozie/thinking-agent.git
cd thinking-agent
```

**2. Install all workspace dependencies**
```bash
uv sync
```

**3. Set up environment variables**

Create `mcp_server/.env`:
```
TAVILY_API_KEY=your_tavily_key_here
```

Create `agent_client/.env`:
```
OPENROUTER_API_KEY=your_openrouter_key_here
```

Optionally override server host/port (defaults: `0.0.0.0:8000`):
```
MCP_HOST=0.0.0.0
MCP_PORT=8000
```

---

## Running the System

You need **two terminal windows** open simultaneously. Always start the server first.

**Terminal 1 — Start the MCP Server:**
```bash
uv run --package mcp_server python mcp_server/main.py
```

Expected output:
```
[2026-06-01 10:00:00] [SERVER] [INFO] Starting Agricultural Advisory System MCP Server on streamable-http at 0.0.0.0:8000...
```

**Terminal 2 — Start the Agent Client:**
```bash
uv run --package agent_client python agent_client/main.py
```

You will be prompted:
```
Enter your agricultural question:
```

---

## Example Queries

**Queries likely to hit the internal knowledge base:**
```
How do I improve soil health on my farm?
What is the best irrigation strategy for dry regions?
How do I protect my crops from pests without chemicals?
What crops should I grow in a low-rainfall area?
How do I store grain properly after harvest?
```

**Queries that will trigger the Tavily fallback:**
```
What are the latest government agricultural subsidy programs in Nigeria?
What are the newest drone technologies being used in precision farming?
What is the current global price of wheat?
```

---

## Sample Log Output

```
[2026-06-01 16:53:03] [CLIENT] [INFO] Agent started. User query: 'What is the ideal soil pH range for growing maize...'
[2026-06-01 16:53:03] [CLIENT] [INFO] MCP session initialised — sampling handler and server log handler registered
[2026-06-01 16:53:03] [CLIENT] [INFO] Agent ready. Sending query...
[2026-06-01 16:53:05] [CLIENT] [INFO] Querying CRAG resource with: 'ideal soil pH range for growing maize...'
[2026-06-01 16:53:05] [SERVER] [INFO] Expanded query into 5 search vectors
[2026-06-01 16:53:05] [SERVER] [INFO] Level-1 search: 3 candidate documents (top_k=3, corpus=5)
[2026-06-01 16:53:05] [SERVER] [INFO] Level-2 search: 8 chunks with similarity > 0
[2026-06-01 16:53:05] [SERVER] [INFO] Initiating ToT Evaluation on retrieved chunks via LLM sampling...
[2026-06-01 16:53:06] [SERVER] [INFO] ToT [doc_1] analytical=8 contextual=7 practical=6 avg=7.0
[2026-06-01 16:53:06] [SERVER] [INFO] ToT kept 3/8 chunks (threshold=5)
[2026-06-01 16:53:06] [CLIENT] [INFO] CRAG resource response received
[2026-06-01 16:53:11] [CLIENT] [INFO] Calling remote reflection tool on MCP server...
[2026-06-01 16:53:11] [SERVER] [INFO] Initiating critique loop via MCP Sampling...
[2026-06-01 16:53:11] [CLIENT] [INFO] MCP Sampling request received from server, executing LLM locally...
[2026-06-01 16:53:17] [SERVER] [INFO] Critique validated — overall quality: fair
[2026-06-01 16:53:17] [SERVER] [INFO] Initiating correction loop via MCP Sampling...
[2026-06-01 16:53:17] [CLIENT] [INFO] MCP Sampling request received from server, executing LLM locally...
[2026-06-01 16:53:26] [SERVER] [INFO] Correction validated — 6 changes made
[2026-06-01 16:53:26] [CLIENT] [INFO] Reflection tool response received
```

---

## Tech Stack

| Component | Technology |
|---|---|
| MCP Server | FastMCP 3.3.1 |
| Agent Framework | LangChain + LangGraph |
| MCP Client | FastMCP Client |
| LLM Provider | OpenRouter (GPT-4o-mini) |
| Web Fallback | Tavily Search API |
| Dependency Management | uv workspace |
| Transport | streamable-http |
| Data Validation | Pydantic v2 |
| Language | Python 3.14 |

---

## Key Design Decisions

**No API keys on the server** — the MCP server holds no LLM credentials. All model inference is delegated to the client via MCP Sampling (`ctx.sample()`). The server constructs fully-rendered prompts with runtime values and the client executes them against its local LLM.

**Two-level hierarchical retrieval** — Level 1 scores and ranks documents by summary relevance with a `top_k` cap to bound search cost. Level 2 applies Jaccard similarity at the chunk level within only the promoted documents. This prevents a flat scan of the entire corpus.

**Genuine LLM-based ToT** — each retrieved chunk is independently evaluated by the LLM across 3 reasoning paths (Analytical, Contextual, Practical) via `ctx.sample()`. The LLM's structured response is validated by `ToTScoreResponse` before the score is trusted. This is true Tree-of-Thought: the LLM reasons, not a formula.

**Pydantic validation on all sampling responses** — `CritiqueResponse`, `CorrectionResponse`, and `ToTScoreResponse` enforce schema compliance on every LLM output. Validation failures degrade gracefully with a logged warning rather than crashing.

**Single FastMCP client** — one `fastmcp.Client` instance registers both the `sampling_handler` and `log_handler`, ensuring server log messages are actually captured and written to `agent_system.log`. The previous dual-client approach left server logs uncaptured.

**Configurable host/port** — the server reads `MCP_HOST` and `MCP_PORT` from environment variables, making it deployable beyond localhost without code changes.

---

## Repository

Built for KodeCamp Stage 2 — Decentralized Tooling via the Model Context Protocol.
