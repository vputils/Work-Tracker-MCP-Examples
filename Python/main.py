import os
import asyncio

from llm_providers import LLMProvider
from work_tracker_mcp_agent import WorkTrackerMCPAgent


from dotenv import load_dotenv

load_dotenv()

# DEBUG: Uncomment this line to enable detailed DEBUG outputs of the agent graph
# from langchain_core.globals import set_debug; set_debug(True)


# Feel free to modify these
DEFAULT_CLOUD_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_LOCAL_CHAT_MODEL = "qwen3-coder-next"

LLM_PROVIDER_ENV_KEY = "LLM_PROVIDER"
LLM_PROVIDER_DEFAULT = LLMProvider.VLLM.value  # Feel free to modify this

AZURE_AI_MODEL_ENV_KEY = "AZURE_AI_MODEL"
AZURE_AI_MODEL_DEFAULT = DEFAULT_CLOUD_CHAT_MODEL

OPEN_AI_MODEL_ENV_KEY = "OPEN_AI_MODEL"
OPEN_AI_MODEL_DEFAULT = DEFAULT_CLOUD_CHAT_MODEL

VLLM_MODEL_ENV_KEY = "VLLM_MODEL"
VLLM_MODEL_DEFAULT = DEFAULT_LOCAL_CHAT_MODEL


MCP_SEVER_URL_ENV_KEY = "MCP_SERVER_URL"
MCP_SEVER_URL_DEFAULT = "http://localhost:8484/mcp"


async def main():
    stop_event = asyncio.Event()

    llm_provider_from_env = LLMProvider(os.getenv(LLM_PROVIDER_ENV_KEY, LLM_PROVIDER_DEFAULT))
    mcp_server_url = os.getenv(MCP_SEVER_URL_ENV_KEY, MCP_SEVER_URL_DEFAULT)

    match llm_provider_from_env:
        case LLMProvider.OPENAI:
            chat_model = os.getenv(OPEN_AI_MODEL_ENV_KEY, OPEN_AI_MODEL_DEFAULT)
        case LLMProvider.AI_FOUNDRY:
            chat_model = os.getenv(AZURE_AI_MODEL_ENV_KEY, AZURE_AI_MODEL_DEFAULT)
        case LLMProvider.VLLM:
            chat_model = os.getenv(VLLM_MODEL_ENV_KEY, VLLM_MODEL_DEFAULT)
        case _:
            raise ValueError(f"Invalid or not yet implemented LLM provider '{llm_provider_from_env.value}'!")

    try:
        # Feel free to change the LLM provider to any other.
        agent = await WorkTrackerMCPAgent.create(llm_provider_from_env, chat_model, mcp_server_url)
        await agent.run_streaming_loop(stop_event.is_set)
    except KeyboardInterrupt:
        stop_event.set()


# --- The execution ---
if __name__ == "__main__":
    asyncio.run(main())
    
    print("\n\nGood bye!")
