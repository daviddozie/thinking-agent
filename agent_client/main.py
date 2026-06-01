import asyncio
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.context import RequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    TextContent,
    LoggingMessageNotificationParams,
)

load_dotenv()


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

server_logger = logging.getLogger("mcp_server_logs")
server_logger.setLevel(logging.DEBUG)
server_formatter = logging.Formatter(
    "[%(asctime)s] [SERVER] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
server_console = logging.StreamHandler()
server_console.setFormatter(server_formatter)
server_file = logging.FileHandler("agent_system.log")
server_file.setFormatter(server_formatter)
server_logger.addHandler(server_console)
server_logger.addHandler(server_file)


llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3,
)


async def sampling_handler(
    ctx: RequestContext,
    params: CreateMessageRequestParams,
) -> CreateMessageResult:
    logger.info("MCP Sampling request received from server, executing LLM locally...")

    # Extract the prompt text from the MCP sampling message
    prompt_text = ""
    for msg in params.messages:
        if hasattr(msg.content, "text"):
            prompt_text += msg.content.text + "\n"

    # Run the LLM locally on the client
    response = await llm.ainvoke(prompt_text)
    result_text = response.content

    logger.info("MCP Sampling complete, returning result to server")

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=result_text),
        model="gpt-4o-mini",
        stopReason="endTurn",
    )


async def server_log_handler(
    params: LoggingMessageNotificationParams,
    context,
) -> None:
    level = str(params.level).upper()
    message = params.data if isinstance(params.data, str) else str(params.data)

    if level == "ERROR":
        server_logger.error(message)
    elif level == "WARNING":
        server_logger.warning(message)
    elif level == "DEBUG":
        server_logger.debug(message)
    else:
        server_logger.info(message)



async def run_agent(user_query: str):
    logger.info(f"Agent started. User query: '{user_query}'")
    callbacks = Callbacks(on_logging_message=server_log_handler)

    # Initialize MCP client with callbacks
    mcp_client = MultiServerMCPClient(
        connections={
            "agri_server": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        },
        callbacks=callbacks,
    )

    logger.info("Connecting to MCP server with sampling support...")

    async with streamable_http_client("http://localhost:8000/mcp") as (read, write, _):
        async with ClientSession(
            read,
            write,
            sampling_callback=sampling_handler,
        ) as session:
            await session.initialize()
            logger.info("MCP session initialized with sampling handler registered")

            # Fetch tools via the mcp_client (uses separate connections per call)
            mcp_tools = await mcp_client.get_tools()
            logger.info(f"Fetched {len(mcp_tools)} tools from MCP server")

            @tool
            async def reflection_tool(original_query: str, draft_answer: str) -> str:
                """
                Critiques and corrects a draft answer using true MCP Sampling.
                The server calls ctx.sample() which fires our sampling_handler,
                runs the LLM locally on the client, and returns the result.
                """
                logger.info("Calling remote reflection tool on MCP server...")

                # Call the tool directly via the sampling-enabled session
                result = await session.call_tool(
                    "reflect_on_answer",
                    arguments={
                        "original_query": original_query,
                        "draft_answer": draft_answer,
                    },
                )
                logger.info("Reflection tool response received")
                if result.content:
                    return result.content[0].text
                return "No reflection result"

            @tool
            async def crag_knowledge_tool(query: str) -> str:
                """
                Queries the agricultural knowledge base on the MCP server.
                Uses hierarchical search + Tree-of-Thought + Tavily fallback.
                """
                logger.info(f"Querying CRAG resource with: '{query}'")
                try:
                    result = await session.read_resource(
                        f"knowledge://agriculture/docs/{query}"
                    )
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
3. ALWAYS call reflection_tool with your original query and draft answer
4. Return the final corrected answer from the reflection tool
"""

            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_prompt,
            )

            logger.info("Agent ready. Sending query...")

            try:
                response = await agent.ainvoke(
                    {
                        "messages": [{"role": "user", "content": user_query}],
                    },
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
