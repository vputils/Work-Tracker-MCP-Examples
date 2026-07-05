import 'dotenv/config';
import { LLMProvider } from './llmProviders.js';
import { WorkTrackerMCPAgent } from './workTrackerMcpAgent.js';

// DEBUG: Uncomment this line to enable detailed DEBUG outputs of the agent graph
// process.env.LANGCHAIN_VERBOSE = "true";

// Feel free to modify these
const DEFAULT_CLOUD_CHAT_MODEL = "gpt-4.1-mini";
const DEFAULT_LOCAL_CHAT_MODEL = "qwen3-coder-next";

const LLM_PROVIDER_ENV_KEY = "LLM_PROVIDER";
const LLM_PROVIDER_DEFAULT = LLMProvider.VLLM; // Feel free to modify this

const AZURE_AI_MODEL_ENV_KEY = "AZURE_AI_MODEL";
const AZURE_AI_MODEL_DEFAULT = DEFAULT_CLOUD_CHAT_MODEL;

const OPEN_AI_MODEL_ENV_KEY = "OPEN_AI_MODEL";
const OPEN_AI_MODEL_DEFAULT = DEFAULT_CLOUD_CHAT_MODEL;

const VLLM_MODEL_ENV_KEY = "VLLM_MODEL";
const VLLM_MODEL_DEFAULT = DEFAULT_LOCAL_CHAT_MODEL;

const MCP_SERVER_URL_ENV_KEY = "MCP_SERVER_URL";
const MCP_SERVER_URL_DEFAULT = "http://localhost:8484/mcp";


async function main() {
    let terminate = false;

    // Handle graceful shutdown (equivalent to catching KeyboardInterrupt in Python)
    process.on('SIGINT', () => {
        terminate = true;
    });

    const llmProviderFromEnv = (process.env[LLM_PROVIDER_ENV_KEY] as LLMProvider) || LLM_PROVIDER_DEFAULT;
    const mcpServerUrl = process.env[MCP_SERVER_URL_ENV_KEY] || MCP_SERVER_URL_DEFAULT;

    let chatModel: string;

    switch (llmProviderFromEnv) {
        case LLMProvider.OPENAI:
            chatModel = process.env[OPEN_AI_MODEL_ENV_KEY] || OPEN_AI_MODEL_DEFAULT;
            break;
        case LLMProvider.AI_FOUNDRY:
            chatModel = process.env[AZURE_AI_MODEL_ENV_KEY] || AZURE_AI_MODEL_DEFAULT;
            break;
        case LLMProvider.VLLM:
            chatModel = process.env[VLLM_MODEL_ENV_KEY] || VLLM_MODEL_DEFAULT;
            break;
        default:
            throw new Error(`Invalid or not yet implemented LLM provider ${llmProviderFromEnv}!`);
    }

    try {
        const agent = await WorkTrackerMCPAgent.create(llmProviderFromEnv, chatModel, mcpServerUrl);
        await agent.runStreamingLoop(() => terminate);
    } catch (error) {
        // We catch errors just in case something fails gracefully during shutdown or init
        if (!terminate) {
            console.error("\nAgent execution failed:", error);
        }
    }
}

// --- The execution ---
if (require.main === module) {
    main().then(() => {
        console.log("\n\nGood bye!");
        process.exit(0);
    });
}