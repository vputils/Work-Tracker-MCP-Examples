import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { ChatOpenAI, AzureChatOpenAI } from "@langchain/openai";
import { DefaultAzureCredential, getBearerTokenProvider } from "@azure/identity";

const AZURE_AI_RESOURCE_NAME_ENV_KEY = "AZURE_AI_RESOURCE_NAME";
const AZURE_AI_CHAT_DEPLOYMENT_ENV_KEY = "AZURE_AI_CHAT_DEPLOYMENT";
const AZURE_AI_API_VERSION_ENV_KEY = "AZURE_AI_API_VERSION";
const AZURE_AI_API_VERSION_DEFAULT = "2024-04-01-preview";

const VLLM_API_BASE_ENV_KEY = "VLLM_API_BASE";
const VLLM_API_BASE_DEFAULT = "http://localhost:8090/v1";
const VLLM_API_KEY_ENV_KEY = "VLLM_API_KEY";
const VLLM_API_KEY_DEFAULT = "EMPTY"; // By default, the local VLLM does not require an API key

const OPENAI_API_KEY_ENV_KEY = "OPENAI_API_KEY"; 

/**
 * The enum of supported LLM providers.
 */
export enum LLMProvider {
    OPENAI = "openai",
    AI_FOUNDRY = "ai_foundry",
    VLLM = "vllm"
}

export function initModelProvider(provider: LLMProvider, model: string, temperature: number): BaseChatModel {
    switch (provider) {
        case LLMProvider.OPENAI:
            return _initOpenAI(model, temperature);
        case LLMProvider.AI_FOUNDRY:
            return _initAIFoundry(model, temperature);
        case LLMProvider.VLLM:
            return _initVLLM(model, temperature);
        default:
            throw new Error(`Invalid or not yet implemented LLM provider '${provider}'!`);
    }
}

function _initOpenAI(model: string, temperature: number): BaseChatModel {
    if (!process.env[OPENAI_API_KEY_ENV_KEY]) {
        throw Error(
            `You must set the '${OPENAI_API_KEY_ENV_KEY}' environment variable (or put it into the \`.env\` file) when instantiating an OpenAI model!`
        );
    }

    // The API key is automatically picked up from process.env["OPENAI_API_KEY"] by default
    return new ChatOpenAI({ model, temperature })
}

function _initAIFoundry(model: string, temperature: number): BaseChatModel {
    // Inits using Azure CLI globally saved credentials - use `az login` command to set them.
    if (!process.env[AZURE_AI_RESOURCE_NAME_ENV_KEY]) {
        throw Error(`You must set the '${AZURE_AI_RESOURCE_NAME_ENV_KEY}' environment variable (or put it into the \`.env\` file) when instantiating an AI Foundry model!`);
    }

    const credential = new DefaultAzureCredential();
    const azureADTokenProvider = getBearerTokenProvider(
        credential,
        "https://cognitiveservices.azure.com/.default"
    );

    // In manual deployments, the deployment name tends to correspond to the model name.
    const deploymentName = process.env[AZURE_AI_CHAT_DEPLOYMENT_ENV_KEY] || model;
    const resourceName = process.env[AZURE_AI_RESOURCE_NAME_ENV_KEY];
    const apiVersion = process.env[AZURE_AI_API_VERSION_ENV_KEY] || AZURE_AI_API_VERSION_DEFAULT;
    
    return new AzureChatOpenAI({
        azureOpenAIApiDeploymentName: deploymentName, 
        azureOpenAIApiInstanceName: resourceName,
        azureOpenAIApiVersion: apiVersion, 
        azureADTokenProvider,
        temperature,
    });
}

function _initVLLM(model: string, temperature: number): BaseChatModel {
    const apiKey = process.env[VLLM_API_KEY_ENV_KEY] || VLLM_API_KEY_DEFAULT;
    const baseURL = process.env[VLLM_API_BASE_ENV_KEY] || VLLM_API_BASE_DEFAULT;

    return new ChatOpenAI({ 
        model, temperature, apiKey,
        configuration: { baseURL },
    })
}