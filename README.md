# Work Tracker MCP Examples

Example agentic workflows that connect to the Model Context Protocol (MCP) server exposed by the desktop version of **Work Tracker: Hours & ManDays**, showing how the app's time-tracking data and actions can be driven by a custom AI agent instead of the app's own UI.

> **Note on how this repository was built.** The code in this repository was written and reviewed by hand, not vibe-coded. Every module was designed deliberately (state graph, tool registry, provider abstraction, guardrails) and is meant to be read as a reference implementation, not as throwaway output from prompting a model until something ran. This repository is also part of the author's public portfolio, so the code is held to that standard.

## What is Work Tracker: Hours & ManDays

[Work Tracker: Hours & ManDays](https://www.vputils.com) is a fast, lightweight, and effortless time-tracking application by VP Utils, built for mobile, desktop, and web.

Key characteristics of the app, as described on its landing page:

- **Privacy-first design** - data stays on-device by default; a Pro subscription adds RSA-2048 encrypted cloud sync.
- **Hours and Man-Days tracking** - switch between tracking modes with configurable time rounding.
- **Workspaces (Spaces)** - Pro subscribers can organize work across multiple project workspaces, each with its own settings and history, with automatic timer pausing when switching between projects.
- **Multi-platform** - available on iOS, Android, desktop (Windows, macOS, Linux), and web.
- **Deadlines and live notifications** - task deadline alerts and lock-screen/Dynamic Island activity display.
- **Data export** - Premium subscribers can export data to CSV and XLSX.
- **AI MCP Server** - the desktop version can expose a Model Context Protocol server, letting AI assistants and custom agents read and manage the user's work data directly. This is the feature demonstrated in this repository.

### Download Work Tracker

The buttons below mirror the download section of [www.vputils.com](https://www.vputils.com), using each platform's own official badge artwork:

<p align="center">
  <a href="https://apps.apple.com/app/work-tracker-hours-mandays/id6758014785"><img alt="Download on the App Store" src="https://developer.apple.com/assets/elements/badges/download-on-the-app-store.svg" height="48"></a>
  &nbsp;&nbsp;
  <a href="https://play.google.com/store/apps/details?id=com.vputils.work_tracker"><img alt="Get it on Google Play" src="https://cdn.jsdelivr.net/gh/pioug/google-play-badges@master/svg/en.svg" height="48"></a>
  &nbsp;&nbsp;
  <a href="https://www.vputils.com/WebApp"><img alt="Launch Web App" src="assets/web-app-badge.svg" height="48"></a>
</p>
<p align="center">
  <a href="https://apps.apple.com/us/app/work-tracker-hours-mandays/id6758014785"><img alt="Download on the Mac App Store" src="https://tools.applemediaservices.com/api/badges/download-on-the-mac-app-store/black/en-us?size=250x83" height="48"></a>
  &nbsp;&nbsp;
  <a href="https://apps.microsoft.com/detail/9NPVTWGLXD5G"><img alt="Get it from Microsoft" src="https://get.microsoft.com/images/en-us%20dark.svg" height="48"></a>
  &nbsp;&nbsp;
  <a href="https://snapcraft.io/work-tracker"><img alt="Get it from the Snap Store" src="https://snapcraft.io/en/dark/install.svg" height="48"></a>
</p>

A direct AppImage download for Linux (x64) is also available: [AppImage (x64)](https://www.vputils.com/api/desktop-binaries/download/1/linux/x64)

Full landing page, features, and pricing: [https://www.vputils.com](https://www.vputils.com)

## What this repository demonstrates

The Model Context Protocol (MCP) lets an application expose its data and actions as a set of callable tools over a standard transport, so any MCP-aware client can use them. The desktop build of Work Tracker ships an MCP server (a `StreamableHTTPServerTransport`, reachable by default at `http://localhost:8484/mcp`) that exposes tools such as `list_spaces`, `list_actionable_tasks`, `get_current_work_status`, `start_work_day`, `start_task_work`, `get_time_report`, `search_tasks`, `delete_task`, and `delete_work_day`.

This repository shows how to build a custom agent, independent of the Work Tracker app's own UI, that:

1. Connects to that MCP server and auto-discovers every tool it exposes, instead of hard-coding a fixed tool list.
2. Combines the discovered remote tools with locally defined tools (for example, resolving relative time expressions or exporting a Markdown report to disk).
3. Runs a LangGraph agent loop on top of those combined tools, with guardrails such as a maximum number of simultaneous tool calls and a circuit breaker on the number of tool-calling rounds.
4. Talks to any of several LLM providers (OpenAI, Azure AI Foundry, or a local OpenAI-compatible vLLM endpoint) through one small abstraction, so the same agent logic runs against a hosted or a fully local model.

The intent is for developers to use this as a starting point for their own agentic workflows against Work Tracker's MCP server, whether that means a different orchestration framework, a different LLM provider, or additional local tools.

## Repository structure

```
Work-Tracker-MCP-Examples/
├── LICENSE
├── README.md
├── assets/
│   └── web-app-badge.svg          # Badge used for the "Launch Web App" button above
├── Python/                        # Python example (LangGraph + langchain-mcp-adapters)
│   ├── main.py                     # Entry point: loads config, starts the agent's chat loop
│   ├── work_tracker_mcp_agent.py   # The WorkTrackerMCPAgent class and its LangGraph graph
│   ├── agent_tools.py              # Local (non-MCP) tools available to the agent
│   ├── llm_providers.py            # LLM provider abstraction (OpenAI / Azure AI Foundry / vLLM)
│   ├── requirements.txt
│   └── .env.example
└── TypeScript/                    # TypeScript example (LangGraph.js + the official MCP SDK)
    ├── src/
    │   ├── main.ts                  # Entry point: loads config, starts the agent's chat loop
    │   ├── workTrackerMcpAgent.ts    # The WorkTrackerMCPAgent class and its LangGraph graph
    │   ├── agentTools.ts            # Local (non-MCP) tools available to the agent
    │   ├── llmProviders.ts          # LLM provider abstraction (OpenAI / Azure AI Foundry / vLLM)
    │   └── utils.ts                 # Small helpers (home-directory path expansion)
    ├── package.json
    ├── tsconfig.json
    └── .env.example
```

The Python and TypeScript examples are two independent, functionally equivalent implementations of the same agent: same MCP discovery flow, same local tools, same LangGraph state machine shape, and the same system prompt and guardrails. A few implementation details differ where each ecosystem's idioms differ; those are called out inline below. Pick whichever matches your stack.

## Getting the code

```bash
git clone https://github.com/vputils/Work-Tracker-MCP-Examples.git
cd Work-Tracker-MCP-Examples
```

Then follow the setup for whichever example you want to run: [Python](#python-example) or [TypeScript](#typescript-example).

## Python example

### How it works

#### `llm_providers.py`

Defines the `LLMProvider` enum (`openai`, `ai_foundry`, `vllm`) and `init_model_provider()`, a single factory function that returns a LangChain `BaseChatModel` configured for whichever provider is selected:

- `openai` - `ChatOpenAI`, requires `OPENAI_API_KEY`.
- `ai_foundry` - `AzureAIOpenAIApiChatModel`, authenticated via `DefaultAzureCredential` (Azure CLI login), requires `AZURE_AI_PROJECT_ENDPOINT`.
- `vllm` - `ChatOpenAI` pointed at a local OpenAI-compatible vLLM server, so no cloud API key is needed by default.

#### `agent_tools.py`

Defines the local tools that run outside the Work Tracker MCP server, exposed via `get_all_local_agent_tools()`:

- `get_relative_timestamp_ms` - converts a human-readable relative time expression (for example "yesterday at 8am") into a UTC millisecond timestamp, using `dateparser`. The agent is instructed to call this before any MCP tool that expects a timestamp.
- `export_markdown_report` - writes a Markdown report to `./outputs/WorkTrackerReports`, with filename sanitization to prevent path traversal.
- `read_local_notes` - reads a local text or Markdown file (for example meeting notes or a to-do list) so its content can be turned into tasks.

#### `work_tracker_mcp_agent.py`

The core of the example. `WorkTrackerMCPAgent` is constructed through the async `create()` factory (not the constructor directly), which:

1. Initializes the chosen chat model via `llm_providers.init_model_provider`.
2. Connects to the Work Tracker MCP server through `langchain_mcp_adapters.client.MultiServerMCPClient` and discovers all remote tools with `get_tools()`.
3. Merges the discovered MCP tools with the local tools from `agent_tools.py` into a single tool registry, and binds all of them to the chat model.
4. Builds and compiles a LangGraph `StateGraph` with an in-memory checkpointer, so a conversation can be resumed by reusing the same thread ID.

The graph itself is a small loop:

- `agent` node - prepends a system prompt describing the assistant's role and operational rules (check work-day status before starting a task, be careful with destructive actions, resolve task names to IDs via search instead of inventing IDs, and so on), then invokes the LLM with the full message history.
- `tools` node - executes every tool call requested by the LLM concurrently with `asyncio.gather`, mapping results (or errors) back to `ToolMessage`s.
- A conditional edge routes back to `tools` whenever the LLM's last response contains tool calls, and to `END` otherwise.
- A circuit breaker (`_MAX_TOOL_CALLS_ROUNDS`, default 10) forces the loop to end and hand control back to the user if the agent tries to keep calling tools indefinitely. A second guardrail (`_MAX_SIMULTANEOUS_TOOL_CALLS`, default 5) rejects a single LLM turn that tries to call too many tools at once.

Two entry points are exposed for driving the graph: `run_single_query()` for a single request/response exchange, and `run_streaming_loop()`, which runs an interactive CLI chat loop and streams the assistant's tokens as they are generated.

#### `main.py`

Loads environment variables from `.env`, resolves the LLM provider and model name (with sensible defaults for cloud and local models), constructs the `WorkTrackerMCPAgent`, and starts the streaming CLI chat loop. Terminate the loop with Ctrl+C while the agent is thinking or answering, or with Ctrl+D while it is waiting for input.

### Prerequisites

- Python 3.12 or later (the code uses `StrEnum` and other modern typing features).
- The desktop version of Work Tracker: Hours & ManDays, running locally with its MCP server enabled (default endpoint `http://localhost:8484/mcp`).
- One of the following, depending on the LLM provider you choose:
  - An OpenAI API key, or
  - An Azure AI Foundry project with the Azure CLI logged in (`az login`), or
  - A local OpenAI-compatible vLLM server (default `http://localhost:8090/v1`).

### Setup

1. From the repository root, move into the Python example directory:

   ```bash
   cd Python
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file and fill in the values for your chosen LLM provider:

   ```bash
   cp .env.example .env
   ```

   Relevant variables in `.env`:

   | Variable | Purpose | Default |
   |---|---|---|
   | `LLM_PROVIDER` | Which provider to use: `openai`, `ai_foundry`, or `vllm` | `vllm` |
   | `OPENAI_API_KEY` | Required if `LLM_PROVIDER=openai` | - |
   | `OPEN_AI_MODEL` | Model name for OpenAI | `gpt-4.1-mini` |
   | `AZURE_AI_PROJECT_ENDPOINT` | Required if `LLM_PROVIDER=ai_foundry` | - |
   | `AZURE_AI_MODEL` | Model name for Azure AI Foundry | `gpt-4.1-mini` |
   | `VLLM_API_BASE` | Base URL of a local OpenAI-compatible vLLM server | `http://localhost:8090/v1` |
   | `VLLM_API_KEY` | API key for the vLLM server, if any | `EMPTY` |
   | `VLLM_MODEL` | Model name served by vLLM | `qwen3-coder-next` |
   | `MCP_SERVER_URL` | URL of the Work Tracker desktop app's MCP server | `http://localhost:8484/mcp` |

5. Start the desktop Work Tracker app and make sure its MCP server is running and reachable at the URL configured above.

### Usage

Run the agent's interactive chat loop:

```bash
python main.py
```

On startup, the agent connects to the Work Tracker MCP server, discovers its tools, and reports how many remote and local tools it found. You can then type natural-language requests, for example:

```
User:      Start my work day and begin working on the "Client onboarding" task.
User:      How many hours have I logged this week? Export a summary as a markdown report.
User:      Read ~/notes/standup.md and create tasks from anything that looks actionable.
```

Press Ctrl+C while the agent is thinking or answering, or Ctrl+D while it is waiting for input, to end the session.

## TypeScript example

### How it works

#### `llmProviders.ts`

The TypeScript equivalent of `llm_providers.py`: an `LLMProvider` enum and `initModelProvider()`, returning a LangChain.js `BaseChatModel`:

- `openai` - `ChatOpenAI` from `@langchain/openai`, requires `OPENAI_API_KEY`.
- `ai_foundry` - `AzureChatOpenAI` from `@langchain/openai`, authenticated with a bearer token provider built from `DefaultAzureCredential` (Azure CLI login). This differs from the Python example, which uses the `langchain-azure-ai` package's `AzureAIOpenAIApiChatModel` configured with a project endpoint; the TypeScript version instead targets a specific Azure OpenAI resource and deployment via `AZURE_AI_RESOURCE_NAME` and `AZURE_AI_CHAT_DEPLOYMENT`.
- `vllm` - `ChatOpenAI` pointed at a local OpenAI-compatible vLLM server, so no cloud API key is needed by default.

#### `agentTools.ts`

The TypeScript equivalent of `agent_tools.py`, exposed via `getAllLocalAgentTools()`, with each tool's schema defined using `zod` instead of Python type hints:

- `getRelativeTimestampMs` - the same relative-time-to-UTC-millisecond-timestamp tool, using `chrono-node` in place of Python's `dateparser`.
- `exportMarkdownReport` - writes a Markdown report with the same filename sanitization logic. It defaults to `~/WorkTrackerReports` (the user's home directory) rather than the Python example's `./outputs/WorkTrackerReports` (relative to the working directory); this is an intentional difference between the two examples, not a bug.
- `readLocalNotes` - reads a local text or Markdown file, with the same error handling for missing or unreadable files.

#### `utils.ts`

A small helper module holding `expandHomeDir()`, which expands a leading `~` to the user's home directory. Node.js has no built-in equivalent of Python's `os.path.expanduser`, so this example provides its own.

#### `workTrackerMcpAgent.ts`

The TypeScript equivalent of `work_tracker_mcp_agent.py`, with the same overall shape but a few ecosystem-specific differences:

- It connects to the MCP server directly with the official `@modelcontextprotocol/sdk`'s `Client` and `StreamableHTTPClientTransport`, since there is no TypeScript equivalent of `langchain-mcp-adapters` used here. Each tool returned by `listTools()` is manually wrapped into a LangChain.js `StructuredTool` via `tool()`, forwarding calls to `mcpClient.callTool()` and passing the tool's raw JSON Schema straight through as its `schema` (LangChain.js accepts a raw JSON Schema directly, with no need to hand-translate it into `zod`).
- Graph state is defined with LangGraph.js's `Annotation.Root`, using `messagesStateReducer` as the equivalent of Python's `add_messages`, and `MemorySaver` as the equivalent of `InMemorySaver`.
- The `tools` node executes tool calls concurrently with `Promise.allSettled` instead of `asyncio.gather(..., return_exceptions=True)`.
- The `agent` and `tools` nodes, the conditional routing, the circuit breaker (10 rounds), the simultaneous tool call cap (5), and the system prompt text are all the same as in the Python example.

#### `main.ts`

Loads `.env`, resolves the LLM provider and model name, constructs the `WorkTrackerMCPAgent`, and starts the streaming CLI chat loop. Termination works slightly differently than in the Python CLI: press Ctrl+C at any time (caught via a `SIGINT` handler, the Node.js equivalent of catching `KeyboardInterrupt`), or type `exit` at the `User:` prompt, since Node's `readline` does not surface Ctrl+D as cleanly as Python's `input()` does.

### Prerequisites

- Node.js 20.6 or later (a current LTS release is recommended); the `--import` flag used to run the example requires at least this version.
- The desktop version of Work Tracker: Hours & ManDays, running locally with its MCP server enabled (default endpoint `http://localhost:8484/mcp`).
- One of the following, depending on the LLM provider you choose:
  - An OpenAI API key, or
  - An Azure AI Foundry (Azure OpenAI) resource with the Azure CLI logged in (`az login`), or
  - A local OpenAI-compatible vLLM server (default `http://localhost:8090/v1`).

### Setup

1. From the repository root, move into the TypeScript example directory:

   ```bash
   cd TypeScript
   ```

2. Install the dependencies:

   ```bash
   npm install
   ```

3. Copy the example environment file and fill in the values for your chosen LLM provider:

   ```bash
   cp .env.example .env
   ```

   Relevant variables in `.env`:

   | Variable | Purpose | Default |
   |---|---|---|
   | `LLM_PROVIDER` | Which provider to use: `openai`, `ai_foundry`, or `vllm` | `vllm` |
   | `OPENAI_API_KEY` | Required if `LLM_PROVIDER=openai` | - |
   | `OPEN_AI_MODEL` | Model name for OpenAI | `gpt-4.1-mini` |
   | `AZURE_AI_RESOURCE_NAME` | Required if `LLM_PROVIDER=ai_foundry`; your Azure OpenAI resource name | - |
   | `AZURE_AI_CHAT_DEPLOYMENT` | Deployment name for Azure AI Foundry | `gpt-4.1-mini` |
   | `AZURE_AI_API_VERSION` | Azure OpenAI API version | `2024-04-01-preview` |
   | `VLLM_API_BASE` | Base URL of a local OpenAI-compatible vLLM server | `http://localhost:8090/v1` |
   | `VLLM_API_KEY` | API key for the vLLM server, if any | `EMPTY` |
   | `VLLM_MODEL` | Model name served by vLLM | `qwen3-coder-next` |
   | `MCP_SERVER_URL` | URL of the Work Tracker desktop app's MCP server | `http://localhost:8484/mcp` |

4. Start the desktop Work Tracker app and make sure its MCP server is running and reachable at the URL configured above.

### Usage

Run the agent's interactive chat loop from inside the `TypeScript` directory:

```bash
node --import tsx src/main.ts
```

This uses `tsx` as an ESM loader so the TypeScript source runs directly, with no separate compile step. On startup, the agent connects to the Work Tracker MCP server, discovers its tools, and reports how many remote and local tools it found. You can then type the same kind of natural-language requests as in the Python example:

```
User:      Start my work day and begin working on the "Client onboarding" task.
User:      How many hours have I logged this week? Export a summary as a markdown report.
User:      Read ~/notes/standup.md and create tasks from anything that looks actionable.
```

Press Ctrl+C at any time, or type `exit` at the `User:` prompt, to end the session.

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.

## Links

- Work Tracker: Hours & ManDays - landing page: [https://www.vputils.com](https://www.vputils.com)
- VP Utils contact: vputils.dev@gmail.com
