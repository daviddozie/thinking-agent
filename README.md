# Thinking Agent — Decentralized MCP Architecture

A production-grade, modular AI agent system built with FastMCP and LangChain. The system splits a monolithic Stage 1 agent into two independent processes communicating over streamable-http via the Model Context Protocol (MCP).

The domain is agricultural advisory — the agent answers farming questions by retrieving grounded knowledge from a hierarchical domain index, evaluating relevance via Tree-of-Thought scoring, and improving its answers through a critique-correction reflection loop.

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
    ├── main.py                 ← LangChain agent + MCP client
    ├── agent_system.log        ← auto-generated persistent log file
    └── .env                    ← OPENROUTER_API_KEY
```

---

## System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│            Agent Client                 │
│  LangChain Agent (create_agent)         │
│  MultiServerMCPClient                   │
│  Local LLM (GPT-4o-mini via OpenRouter) │
│  Logger → agent_system.log             │
└──────────────────┬──────────────────────┘
                   │  streamable-http
                   ▼
┌─────────────────────────────────────────┐
│         MCP Server (FastMCP 3.3.1)      │
│                                         │
│  @resource knowledge://agriculture      │
│           /docs/{query}                 │
│    ├── Multi-query expansion            │
│    ├── Hierarchical index search        │
│    ├── Tree-of-Thought evaluation       │
│    └── Tavily fallback (if needed)      │
│                                         │
│  @tool reflect_on_answer                │
│    ├── Critique prompt construction     │
│    └── Correction prompt construction  │
│         (LLM executed on client)        │
└─────────────────────────────────────────┘
```

---

## How It Works

### CRAG Resource Pipeline
When the agent queries the knowledge resource, the server runs a 4-stage pipeline:

1. **Multi-Query Expansion** — the raw query is expanded into 5 semantic search vectors to improve retrieval coverage
2. **Hierarchical Search** — summaries are matched first, then detailed chunks are retrieved from matched documents
3. **Tree-of-Thought Evaluation** — each chunk is scored across 3 reasoning paths (analytical, creative, critical). Chunks scoring below 4/10 are filtered out
4. **Tavily Fallback** — if no chunks pass the ToT threshold, a live Tavily web search is triggered automatically

### Reflection Tool Pipeline
After the agent drafts an answer, it calls the reflection tool which:

1. Constructs a critique prompt and a correction prompt on the server
2. Returns both prompts to the client as a structured payload
3. The client executes both prompts against its local LLM
4. The critique identifies factual errors and missing points
5. The correction produces an improved final answer addressing the critique

### Dual-Stream Logging
Every action is logged with timestamps and source tags:
- `[CLIENT]` — agent orchestration events
- `[SERVER]` — server-side pipeline events

Both streams are written to `agent_client/agent_system.log`.

---

## Prerequisites

- Python 3.11+
- uv installed → https://astral.sh/uv
- OpenRouter API key → https://openrouter.ai
- Tavily API key → https://tavily.com

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/daviddozie/thinking-agent.git
cd thinking-agent
```

**2. Install all dependencies**
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

---

## Running the System

You need **two terminal windows** open simultaneously.

**Terminal 1 — Start the MCP Server:**
```bash
uv run --package mcp_server python mcp_server/main.py
```

Expected output:
```
[2026-05-27 10:00:00] [SERVER] [INFO] Starting AgriAdvisor MCP Server on streamable-http...
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

**Queries that use internal knowledge base (CRAG pipeline):**
```
How do I improve soil health on my farm?
What is the best irrigation strategy for dry regions?
How do I protect my crops from pests without chemicals?
What crops should I grow in a low-rainfall area?
How do I store grain properly after harvest?
What are the best practices for preventing soil erosion on a sloped farm?
```

**Queries that trigger Tavily fallback (outside knowledge base):**
```
What are the latest government agricultural subsidy programs in Nigeria?
What are the newest drone technologies being used in farming?
What is the current global price of wheat?
```

---

## Sample Log Output

```
[2026-05-27 07:09:34] [CLIENT] [INFO] Agent started. User query: 'What are the best practices for preventing soil erosion on a sloped farm?'
[2026-05-27 07:09:35] [CLIENT] [INFO] Connected to MCP server. Fetched 1 tools
[2026-05-27 07:09:35] [CLIENT] [INFO] Agent ready. Sending query...
[2026-05-27 07:09:37] [CLIENT] [INFO] Querying CRAG resource with: 'best practices for preventing soil erosion on sloped farms'
[2026-05-27 07:09:37] [CLIENT] [INFO] Querying CRAG resource with: 'common mistakes farmers make in preventing soil erosion'
[2026-05-27 07:09:37] [CLIENT] [INFO] CRAG resource response received
[2026-05-27 07:09:37] [CLIENT] [INFO] CRAG resource response received
[2026-05-27 07:09:47] [CLIENT] [INFO] Calling remote reflection tool on MCP server...
[2026-05-27 07:09:47] [CLIENT] [INFO] Executing critique via local LLM...
[2026-05-27 07:10:13] [CLIENT] [INFO] Critique complete
[2026-05-27 07:10:13] [CLIENT] [INFO] Executing correction via local LLM...
[2026-05-27 07:10:42] [CLIENT] [INFO] Correction complete
[2026-05-27 07:11:01] [CLIENT] [INFO] Agent final answer: To effectively prevent soil erosion...
```

---

## Tech Stack

| Component | Technology |
|---|---|
| MCP Server | FastMCP 3.3.1 |
| Agent Framework | LangChain 1.3.1 + LangGraph |
| LLM Provider | OpenRouter (GPT-4o-mini) |
| Web Fallback | Tavily Search API |
| Dependency Management | uv workspace |
| Transport | streamable-http |
| Language | Python 3.14 |

---

## Key Design Decisions

**No API keys on the server** — the MCP server holds no LLM credentials. All model inference is executed on the client side. The server constructs prompts and returns them; the client executes them against its locally initialized LLM context.

**Hierarchical knowledge structure** — the knowledge base is organized as summaries mapping to detailed chunks. Search happens at the summary level first, then drills into chunks. This reduces noise and improves retrieval precision.

**ToT as a relevance filter** — retrieved chunks are scored across three reasoning paths before being returned to the agent. This prevents irrelevant context from polluting the agent's answer generation.

**Tavily as a safety net** — if the internal knowledge base cannot answer a query, the system falls back to live web search automatically. The agent always gets an answer.

---

## Repository

Built for KodeCamp Stage 2 — Decentralized Tooling via the Model Context Protocol.
