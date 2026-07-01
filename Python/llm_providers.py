import os
from enum import StrEnum

from pydantic import SecretStr

from langchain_openai import ChatOpenAI

from azure.identity import DefaultAzureCredential
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

from langchain_core.language_models.chat_models import BaseChatModel


AZURE_AI_PROJECT_ENDPOINT_ENV_KEY = "AZURE_AI_PROJECT_ENDPOINT"

VLLM_API_BASE_ENV_KEY = "VLLM_API_BASE"
VLLM_API_BASE_DEFAULT = "http://localhost:8090/v1"
VLLM_API_KEY_ENV_KEY = "VLLM_API_KEY"
VLLM_API_KEY_DEFAULT = "EMPTY"  # By default, the local VLLM does not require an API key

OPENAI_API_KEY_ENV_KEY = "OPENAI_API_KEY"  # Copied from the docstring of the `ChatOpenAI` class.

class LLMProvider(StrEnum):
    """
    The enum of supported LLM providers.
    """
    OPENAI = "openai"
    AI_FOUNDRY = "ai_foundry"
    VLLM = "vllm"


def init_model_provider(provider: LLMProvider, model: str, temperature: float) -> BaseChatModel:
    match provider:
        case LLMProvider.OPENAI:
            return _init_openai(model, temperature)
        case LLMProvider.AI_FOUNDRY:
            return _init_ai_foundry(model, temperature)
        case LLMProvider.VLLM:
            return _init_vllm(model, temperature)
        case _:
            raise ValueError(f"Invalid or not yet implemented LLM provider!")

def _init_openai(model: str, temperature: float) -> BaseChatModel:
    if OPENAI_API_KEY_ENV_KEY not in os.environ:
        raise ValueError(
            f"You must set the '{OPENAI_API_KEY_ENV_KEY}' environment variable (or put it into the `.env` file) when instantiating an OpenAI model!")
    return ChatOpenAI(
        model=model, 
        temperature=temperature,
    )

def _init_ai_foundry(model: str, temperature: float) -> BaseChatModel:
    # Inits using Azure CLI globally saved credentials - use `az login` command to set them.
    if AZURE_AI_PROJECT_ENDPOINT_ENV_KEY not in os.environ:
        raise ValueError(f"You must set the '{AZURE_AI_PROJECT_ENDPOINT_ENV_KEY}' environment variable (or put it into the `.env` file) when instantiating an AI Foundry model!")
    
    return AzureAIOpenAIApiChatModel(
        project_endpoint=os.environ[AZURE_AI_PROJECT_ENDPOINT_ENV_KEY],
        credential=DefaultAzureCredential(),
        model=model,
        temperature=temperature,
    )

def _init_vllm(model: str, temperature: float) -> BaseChatModel:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=SecretStr(os.getenv(VLLM_API_KEY_ENV_KEY, VLLM_API_KEY_DEFAULT)),
        base_url=os.getenv(VLLM_API_BASE_ENV_KEY, VLLM_API_BASE_DEFAULT)
    )
