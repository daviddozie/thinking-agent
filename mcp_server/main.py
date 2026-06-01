from mcp.types import SamplingMessage, TextContent as MCPTextContent
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from tavily import TavilyClient
from knowledge_base import DOCUMENTS

load_dotenv()

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [SERVER] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# intialise server

mcp = FastMCP(
    "Agricultural Advisory System",
)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# helpers
def expand_query(query: str) -> str:
    """
    Multi-query expansion, generate multiple semantic search
    vectors from the original query to improve retrieval coverage.
    """

    expansions = [
        query,
        f"what are the best practices for {query}",
        f"challenges and solutions related to {query}",
        f"how does {query} affect crop yield and farm productivity",
        f"modern techniques and tools for {query} in agriculture",
    ]
    logger.info(f"Expanded query into {len(expansions)} search vectors")
    return expansions


def hierarchical_search(queries: list[str]) -> list[dict]:
    """
    Hierarchical indexing, first match summaries (high-level),
    then collect the detailed chunks from matched documents.
    """
    matched_docs = []

    for doc in DOCUMENTS:
        summary_lower = doc["summary"].lower()
        for query in queries:
            # check if any query keyword matches the summary
            keywords = [word for word in query.lower().split() if len(word) > 4]
            if any(keyword in summary_lower for keyword in keywords):
                if doc not in matched_docs:
                    matched_docs.append(doc)
                break

    logger.info(f"Hierarchical search found {len(matched_docs)} relevant documents")

    # collect chunks from matched documents
    results = []
    for doc in matched_docs:
        for chunk in doc["chunks"]:
            results.append(
                {
                    "doc_id": doc["id"],
                    "summary": doc["summary"],
                    "chunk": chunk,
                }
            )

    return results


def tot_evaluate(chunks: list[dict], query: str) -> list[dict]:
    """
    Tree-of-Thought evaluation, score each chunk across
    3 reasoning paths and filter out low relevance results.
    Threshold: average score must be >= 4 out of 10.
    """
    logger.info("Initiating ToT Evaluation on retrieved chunks...")
    scored = []

    for item in chunks:
        chunk_lower = item["chunk"].lower()
        query_keywords = [word for word in query.lower().split() if len(word) > 3]

        keyword_hits = sum(1 for keyword in query_keywords if keyword in chunk_lower)
        analytical_score = min(10, keyword_hits * 2 + 4)

        summary_keywords = [
            word for word in item["summary"].lower().split() if len(word) > 4
        ]
        summary_hits = sum(
            1 for keyword in query_keywords if keyword in summary_keywords
        )
        creative_score = min(10, summary_hits * 3 + 2)

        word_count = len(item["chunk"].split())
        critical_score = min(10, word_count // 3)

        avg_score = round((analytical_score + creative_score + critical_score) / 3, 2)

        logger.info(
            f"ToT scores for chunk [{item['doc_id']}]: "
            f"analytical={analytical_score}, creative={creative_score}, "
            f"critical={critical_score}, avg={avg_score}"
        )

        if avg_score >= 4:
            item["tot_score"] = avg_score
            scored.append(item)

    logger.info(f"ToT kept {len(scored)}/{len(chunks)} chunks above threshold")
    return scored


# CRAG Resource
@mcp.resource("knowledge://agriculture/docs/{query}")
async def agricultural_knowledge(query: str) -> str:
    """
    Hierarchical CRAG resource for agricultural domain knowledge.
    Implements: multi-query expansion then hierarchical search then
    Tree-of-Thought evaluation then Tavily fallback if needed.
    """
    logger.info(f"CRAG resource queried with: '{query}'")

    expanded_queries = expand_query(query)
    retrieved_chunks = hierarchical_search(expanded_queries)
    relevant_chunks = tot_evaluate(retrieved_chunks, query)

    # Tavily fallback if ToT filtered everything out
    if not relevant_chunks:
        logger.info("ToT found no relevant chunks, triggering Tavily fallback...")
        try:
            tavily_results = tavily.search(query=f"agriculture {query}", max_results=3)
            fallback_content = "\n\n".join(
                f"[Web Source]: {r['title']}\n{r['content']}"
                for r in tavily_results.get("results", [])
            )
            logger.info("Tavily fallback successful")
            return f"[FALLBACK, Web Results]\n\n{fallback_content}"
        except Exception as e:
            logger.error(f"Tavily fallback failed: {e}")
            return "No relevant knowledge found and web fallback failed."

    # Format and return relevant chunks
    output = []
    for item in relevant_chunks:
        output.append(
            f"[Doc: {item['doc_id']} | Score: {item['tot_score']}]\n"
            f"Summary: {item['summary']}\n"
            f"Detail: {item['chunk']}"
        )

    logger.info(f"CRAG resource returning {len(output)} relevant results")
    return "\n\n".join(output)


# Reflector tool
@mcp.tool()
async def reflect_on_answer(
    original_query: str,
    draft_answer: str,
    ctx: Context,
) -> str:
    """
    Reflection tool, uses true MCP Sampling to delegate LLM
    critique and correction back to the client's model.
    The server holds no API key and makes no direct LLM calls.
    """
    logger.info(f"Reflection tool invoked for query: '{original_query}'")
    await ctx.info(f"Reflection tool invoked for query: '{original_query}'")

    # Critique Loop via MCP Sampling
    logger.info("Requesting critique sample from client LLM via MCP Sampling...")
    await ctx.info("Initiating critique loop via MCP Sampling...")

    critique_result = await ctx.sample(
        messages=[
            SamplingMessage(
                role="user",
                content=MCPTextContent(
                    type="text",
                    text=(
                        f"You are an expert agricultural advisor reviewing an answer.\n\n"
                        f"Original Question: {original_query}\n\n"
                        f"Draft Answer: {draft_answer}\n\n"
                        f"Critique this answer. Identify any factual errors, missing key points, "
                        f"or areas that could be improved for a farmer or agronomist. "
                        f"Be specific and concise."
                    ),
                ),
            )
        ],
        max_tokens=500,
    )
    critique_text = critique_result.text
    logger.info("Critique sample received from client LLM")
    await ctx.info("Critique sample received from client LLM")

    # Correction Loop via MCP Sampling
    logger.info("Requesting correction sample from client LLM via MCP Sampling...")
    await ctx.info("Initiating correction loop via MCP Sampling...")

    correction_result = await ctx.sample(
        messages=[
            SamplingMessage(
                role="user",
                content=MCPTextContent(
                    type="text",
                    text=(
                        f"You are an expert agricultural advisor.\n\n"
                        f"Original Question: {original_query}\n\n"
                        f"Draft Answer: {draft_answer}\n\n"
                        f"Critique: {critique_text}\n\n"
                        f"Now write an improved, corrected final answer that addresses "
                        f"the critique. Be clear, accurate, and practical for a farmer."
                    ),
                ),
            )
        ],
        max_tokens=700,
    )
    corrected_answer = correction_result.text
    logger.info("Correction sample received from client LLM")
    await ctx.info("Correction sample received from client LLM")

    return (
        f"[CRITIQUE]\n{critique_text}\n\n"
        f"[CORRECTED ANSWER]\n{corrected_answer}"
    )

#Run server
def main():
    logger.info("Starting Agricultural Advisory System MCP Server on streamable-http...")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
