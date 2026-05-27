import asyncio
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage

load_dotenv()


# Logging Setup

logger = logging.getLogger("agri_agent")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[%(asctime)s] [CLIENT] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler("agent_system.log")

file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Initialize LLM

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3,
)

# Main Agent Runner
async def run_agent(user_query: str):
    logger.info(f"Agent started. User query: '{user_query}'")

    # Initialize MCP Client

    mcp_client = MultiServerMCPClient(
        {
            "agri_server": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    mcp_tools = await mcp_client.get_tools()
    logger.info(f"Connected to MCP server. Fetched {len(mcp_tools)} tools")

    # Wrap tools

    @tool
    async def reflection_tool(original_query: str, draft_answer: str) -> str:
        """
        Critiques and corrects a draft answer using the MCP server's
        Reflection tool. Executes LLM sampling locally on the client.
        """
        logger.info("Calling remote reflection tool on MCP server...")

        for tool in mcp_tools:
            if tool.name == "reflect_on_answer":
                raw = await tool.ainvoke(
                    {
                        "original_query": original_query,
                        "draft_answer": draft_answer,
                    }
                )

                raw_text = raw[0]["text"] if isinstance(raw, list) else raw

                # Split into critique and correction prompts
                parts = raw_text.split("|||")
                critique_prompt = parts[0].replace("CRITIQUE_PROMPT:", "").strip()
                correction_prompt = parts[1].replace("CORRECTION_PROMPT:", "").strip()

                # Execute critique via local LLM
                logger.info("Executing critique via local LLM...")
                critique = await llm.ainvoke(critique_prompt)
                critique_text = critique.content
                logger.info("Critique complete")

                # Execute correction via local LLM
                logger.info("Executing correction via local LLM...")
                correction = await llm.ainvoke(
                    correction_prompt + f"\n\nCritique to address: {critique_text}"
                )
                corrected_answer = correction.content
                logger.info("Correction complete")

                return (
                    f"[CRITIQUE]\n{critique_text}\n\n"
                    f"[CORRECTED ANSWER]\n{corrected_answer}"
                )

        return "Reflection tool not found on server"

    @tool
    async def crag_knowledge_tool(query: str) -> str:
        """
        Queries the agricultural knowledge base on the MCP server.
        Uses hierarchical search + Tree-of-Thought evaluation + Tavily fallback.
        """
        logger.info(f"Querying CRAG resource with: '{query}'")
        try:
            async with mcp_client.session("agri_server") as session:
                from langchain_mcp_adapters.tools import load_mcp_tools

                uri = f"knowledge://agriculture/docs/{query}"
                result = await session.read_resource(uri)
                logger.info("CRAG resource response received")
                if result and result.contents:
                    return result.contents[0].text
                return "No knowledge found"
        except Exception as e:
            logger.error(f"CRAG resource error: {e}")
            return f"Knowledge retrieval failed: {e}"

    tools = [reflection_tool, crag_knowledge_tool]
    system_prompt = """You are an expert agricultural advisor assistant.

    You MUST always follow these steps in order for EVERY question:
    1. ALWAYS call crag_knowledge_tool first to retrieve relevant knowledge
    2. Use the retrieved knowledge to draft a detailed answer
    3. ALWAYS call reflection_tool with your original query and draft answer to critique and improve it
    4. Return the final corrected answer from the reflection tool

    Never skip the reflection_tool step. It is mandatory for every response."""

    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

    logger.info("Agent ready. Sending query...")

    try:
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_query}]}
        )

        final_answer = response["messages"][-1].content
        logger.info(f"Agent final answer: {final_answer}")
        print(f"\n{'='*60}")
        print(f"FINAL ANSWER:\n{final_answer}")
        print(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise


if __name__ == "__main__":
    query = input("Enter your agricultural question: ")
    asyncio.run(run_agent(query))
