import asyncio
import logging
import os
from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.logging import LogMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    TextContent,
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
    messages: list,
    params: CreateMessageRequestParams,
    ctx,
) -> CreateMessageResult:
    logger.info("MCP Sampling request received from server, executing LLM locally...")

    # Extract the prompt text from the MCP sampling messages
    prompt_text = ""
    for msg in messages:
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


async def server_log_handler(log_message: LogMessage) -> None:
    level = str(log_message.level).upper()
    message = (
        log_message.data
        if isinstance(log_message.data, str)
        else str(log_message.data)
    )

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

    async with Client(
        "http://localhost:8000/mcp",
        sampling_handler=sampling_handler,
        log_handler=server_log_handler,
    ) as client:
        logger.info(
            "MCP session initialised, sampling handler and server log handler registered"
        )

        @tool
        async def crag_knowledge_tool(query: str) -> str:
            """
            Queries the agricultural knowledge base on the MCP server.
            Uses hierarchical search (2-level) + Tree-of-Thought + Tavily fallback.
            """
            logger.info(f"Querying CRAG resource with: '{query}'")
            try:
                results = await client.read_resource(
                    f"knowledge://agriculture/docs/{query}"
                )
                logger.info("CRAG resource response received")
                if results:
                    item = results[0]
                    return item.text if hasattr(item, "text") else item.blob.decode()
                return "No knowledge found"
            except Exception as e:
                logger.error(f"CRAG resource error: {e}")
                return f"Knowledge retrieval failed: {e}"

        @tool
        async def reflection_tool(original_query: str, draft_answer: str) -> str:
            """
            Critiques and corrects a draft answer using true MCP Sampling.
            The server calls ctx.sample() which fires the sampling_handler,
            runs the LLM locally on this client, and returns the validated result.
            """
            logger.info("Calling remote reflection tool on MCP server...")
            result = await client.call_tool(
                "reflect_on_answer",
                arguments={
                    "original_query": original_query,
                    "draft_answer": draft_answer,
                },
            )
            logger.info("Reflection tool response received")
            if result.content:
                block = result.content[0]
                return block.text if hasattr(block, "text") else str(block)
            return "No reflection result"

        tools = [crag_knowledge_tool, reflection_tool]

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
