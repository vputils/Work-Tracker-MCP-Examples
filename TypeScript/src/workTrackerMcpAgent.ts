import { randomUUID } from 'crypto';
import * as readline from 'readline/promises';

import { BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessageChunk } from "@langchain/core/messages";
import { Annotation, END, MemorySaver, messagesStateReducer, START, StateGraph } from "@langchain/langgraph";
import { initModelProvider, LLMProvider } from "./llmProviders";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";

import { Runnable } from "@langchain/core/runnables";
import { StructuredTool, tool } from "@langchain/core/tools";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

import { getAllLocalAgentTools } from "./agentTools.js";
import dedent from 'dedent';


const DEFAULT_CHAT_MODEL_TEMPERATURE = 0.0;

const WorkTrackerMCPAgentStateAnnotation = Annotation.Root({
    // messagesStateReducer acts exactly like `add_messages`
    messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => [],
    }),
    loopCount: Annotation<number>({
        reducer: (oldState, newState) => newState, // Overwrites (default behavior)
        default: () => 0,
    })
});

type WorkTrackerMCPAgentState = typeof WorkTrackerMCPAgentStateAnnotation.State;


export class WorkTrackerMCPAgent {
    public llmProvider: LLMProvider;
    public mcpServerUrl: string;
    public chatModel: string;
    public chatModelTemperature: number;

    private llm: BaseChatModel;
    private llmWithTools!: Runnable<any, any>;
    private graph!: ReturnType<WorkTrackerMCPAgent["buildGraph"]>;
    private memoryCheckpointer: MemorySaver;

    private mcpClient!: Client;

    private discoveredTools!: StructuredTool[];
    private localTools!: StructuredTool[];
    private allTools!: StructuredTool[];    
    private toolsRegistry!: Record<string, StructuredTool>;

    private static readonly MAX_SIMULTANEOUS_TOOL_CALLS = 5;
    private static readonly MAX_TOOL_CALLS_ROUNDS = 10;

    /**
     * The private constructor of the WorkTrackerMCPAgent class.
     * @param llmProvider 
     * @param chatModel 
     * @param mcpServerUrl 
     * @param chatModelTemperature 
     */
    private constructor(
        llmProvider: LLMProvider, 
        chatModel: string, 
        mcpServerUrl: string, 
        chatModelTemperature: number = DEFAULT_CHAT_MODEL_TEMPERATURE
    ) {
        this.chatModel = chatModel;
        this.chatModelTemperature = chatModelTemperature;
        this.llmProvider = llmProvider;
        this.mcpServerUrl = mcpServerUrl;

        this.llm = initModelProvider(this.llmProvider, this.chatModel, this.chatModelTemperature);

        this.memoryCheckpointer = new MemorySaver();
    }
    /**
     * The public factory of the WorkTrackerMCPAgent class.
     * @param llmProvider 
     * @param chatModel 
     * @param mcpServerUrl 
     * @param chatModelTemperature 
     */
    public static async create(
        llmProvider: LLMProvider, 
        chatModel: string, 
        mcpServerUrl: string, 
        chatModelTemperature: number = DEFAULT_CHAT_MODEL_TEMPERATURE
    ): Promise<WorkTrackerMCPAgent> {
        const instance = new WorkTrackerMCPAgent(llmProvider, chatModel, mcpServerUrl, chatModelTemperature);
        await instance.initMCPAndGraph();
        return instance;
    }

    /**
     * Process a single user query.
     * @param userMessage The user query.
     * @param threadId Optional ID of the chat thread to continue on.
     * @returns Promise with the resulting Thread ID, which can be later used to resume the in-memory saved chat thread.
     */
    public async runSingleQuery(userMessage: string, threadId: string | undefined = undefined): Promise<string> {
        threadId ??= randomUUID();
        const config = { configurable: { thread_id: threadId } };

        const initialState: WorkTrackerMCPAgentState = {
            messages: [new HumanMessage(userMessage)],
            loopCount: 0,
        }

        const result = await this.graph.invoke(initialState, config) as WorkTrackerMCPAgentState;

        console.log(`\nAssistant: ${result.messages}`)

        return threadId;
    }

    /**
     * Run a CLI streaming user chat loop.
     * Exit from the loop by pressing Ctrl+C during the agent's thinking or answering,
     * or by pressing Ctrl+D when waiting for a user input.
     * @param cancellationCallback The callback to be called to check if a termination has been requested.
     * @param threadId Optional ID of the chat thread to continue on.
     * @returns Promise with the resulting Thread ID, which can be later used to resume the in-memory saved chat thread.
     */
    public async runStreamingLoop(cancellationCallback: () => boolean, threadId: string | undefined = undefined): Promise<string> {
        threadId ??= randomUUID();
        const config = { configurable: { thread_id: threadId } };

        // Print the initial info to the user
        console.log(dedent(`
            --- Session Started ---
        
            In order to terminate the agent,
            press Ctrl+C when the agent is thinking or answering
            or Ctrl+D when waiting for the user input.
        `).trim() + "\n")

        // Create the readline interface
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        try {
            while (!cancellationCallback()) {
                // Await the user input
                const userMessage = await rl.question("User:      ");

                // Handle Ctrl+D
                // (EOF in Node.js readline often returns an empty string or requires a specific event listener, 
                // but we can also support an explicit exit command)
                if (userMessage.trim().toLowerCase() === 'exit') {
                    break;
                }

                const passedState = {
                    messages: [new HumanMessage(userMessage)],
                    loopCount: 0
                };

                process.stdout.write("\nThinking...\r"); // So that it can be rewritten

                const stream = await this.graph.stream(
                    passedState,
                    { ...config, streamMode: "messages" }
                );

                let prefixPrinted = false;

                for await (const chunk of stream) {
                    if (cancellationCallback()) {
                        break;
                    }

                    const [msg, metadata] = chunk;

                    // We target content chunks originating specifically from our 'agent' node
                    if (metadata.langgraph_node === WorkTrackerMCPAgent.AGENT_NODE_NAME) {
                        // Check if the chunk contains text content (ignores empty tool-call chunks)
                        if (msg.content && typeof msg.content === "string") {
                            if (!prefixPrinted) {
                                // Clear the "Thinking..." line
                                process.stdout.write("\r\x1b[K"); 
                                process.stdout.write("Assistant: ");
                                prefixPrinted = true;
                            }

                            process.stdout.write(msg.content);
                        }
                    }
                }

                if (prefixPrinted) {
                    console.log("\n\n");
                }
            }
        } finally {
            // ALWAYS close the interface in a finally block to prevent memory leaks or hanging processes
            rl.close();
        }

        return threadId;
    }

    // --- MCP Discovery and Graph initialization ---

    /**
     * Asynchronously connects to the Work Tracker MCP server, discovers tools, and builds the graph.
     */
    private async initMCPAndGraph() : Promise<void> {
        // Connect to the MCP server
        console.log(`Connecting to Work Tracker MCP server at '${this.mcpServerUrl}'...`);
        const transport = new StreamableHTTPClientTransport(new URL(this.mcpServerUrl));

        this.mcpClient = new Client(
            { name: "work-tracker-mcp-agent", version: "1.0.0" },
            { capabilities: { } } // The client doesn't need to offer any special capabilities
        );

        await this.mcpClient.connect(transport);

        // Automatically fetch all tools exposed by the MCP server (list_spaces, start_work_day, etc.)
        const mcpToolsResponse = await this.mcpClient.listTools();

        // Translate raw MCP tools into LangChain-compatible tools
        this.discoveredTools = mcpToolsResponse.tools.map(mcpTool => {
            return tool(
                async (args: Record<string, unknown>) => {
                    // When LangGraph calls this tool, we forward the call to the Dart server
                    const result = await this.mcpClient.callTool({
                        name: mcpTool.name,
                        arguments: args
                    });

                    const contentBlocks = result.content as Array<{ type: string; text?: string }>;
                    
                    // MCP returns an array of content blocks; we extract the text
                    // Check if the result has content, and map text properties
                    return contentBlocks
                        .filter(c => c.type === "text")
                        .map(c => c.text)
                        .join("\n");
                },
                {
                    name: mcpTool.name,
                    description: mcpTool.description || `Executes the ${mcpTool.name} action via MCP.`,
                    // LangChain.js allows us to pass the raw JSON Schema
                    // directly from the MCP server without having to parse it into Zod!
                    schema: mcpTool.inputSchema as any, 
                }
            );
        });

        // Combine local tools and remote MCP tools
        this.localTools = getAllLocalAgentTools();
        this.allTools = [...this.discoveredTools, ...this.localTools];

        // Build the tools registry and and bind them to the LLM
        this.toolsRegistry = {};
        for (const t of this.allTools) {
            this.toolsRegistry[t.name] = t;
        }

        this.llmWithTools = this.llm.bindTools!(Object.values(this.toolsRegistry));

        // Now that tools are bound, compile the graph
        this.graph = this.buildGraph();

        console.log(`Successfully connected! Discovered ${this.discoveredTools.length} MCP tools and adding ${this.localTools.length} local tools.`);
    }

    // --- The nodes ---
    private static readonly AGENT_NODE_NAME = "agent" as const;
    private static readonly TOOLS_NODE_NAME = "tools" as const;

    /**
     * Node: Invokes the LLM with the current conversation history, prepending the system message.
     * @param state The input graph state.
     * @returns The resulting graph state partial update.
     */
    private async agentNode(state: WorkTrackerMCPAgentState): Promise<Partial<WorkTrackerMCPAgentState>> {
        const localToolsNames = [];
        for (const t of this.localTools) {
            localToolsNames.push(t.name);
        }
        const toolNamesString = this.localTools.map(tool => tool.name).join("`, `");
        const localToolsAddend = this.localTools.length > 0 ? ` along with a set of local tools (\`${toolNamesString}\`)` : "";

        const systemMessage = dedent(`
            You are a highly capable, integrated executive assistant connected directly to the user's Work Tracker application via the Model Context Protocol (MCP).
            Your primary role is to seamlessly manage their workspaces, tasks, work days, and time logging.

            ### Operational Guidelines:
            1. **Auto-Discovery:** You have access to a suite of Work Tracker tools (e.g., \`list_actionable_tasks\`, \`start_work_day\`, \`start_task_work\`)${localToolsAddend}. 
                Use them proactively to fulfill the user's intent.
            2. **State Dependencies:** If the user asks to "start working on a task", you MUST check if there is an active work day using \`get_current_work_status\`.
                If there isn't one, invoke \`start_work_day\` before invoking \`start_task_work\`.
            3. **Data Retrieval:**
                If asked for a summary, utilize tools like \`get_time_report\` or \`list_work_days\` to provide accurate, up-to-date metrics. 
            4. **Destructive Actions:** Proceed carefully with deletions (\`delete_task\`, \`delete_work_day\`). 
                If the user request is ambiguous, ask for confirmation before executing.
            5. **Efficiency:** You may execute up to ${WorkTrackerMCPAgent.MAX_SIMULTANEOUS_TOOL_CALLS} tool calls simultaneously and before giving an answer to the user, in only at most ${WorkTrackerMCPAgent.MAX_TOOL_CALLS_ROUNDS} rounds. 
                Therefore if you were about to call tools in more than ${WorkTrackerMCPAgent.MAX_TOOL_CALLS_ROUNDS}, answer to the user after at most ${WorkTrackerMCPAgent.MAX_TOOL_CALLS_ROUNDS} tool call rounds and ask the user for a follow-up if they want to proceed.
                Do not invent task IDs; search for them if the user provides a name instead of an ID using \`search_tasks\`.

            Communicate concisely and professionally. You are assisting a power user who values speed and exactness.

            The current date and time is: ${new Date().toLocaleString(undefined, { 
                weekday: 'long', 
                year: 'numeric', 
                month: 'numeric', 
                day: 'numeric', 
                hour: 'numeric', 
                minute: 'numeric', 
                second: 'numeric' 
            })}.
        `).trim();

        const allMessages = [new SystemMessage(systemMessage), ...state.messages];

        // The LLM looks at the history and decides to either reply text or output a tool call.
        const response = await this.llmWithTools.invoke(allMessages);
        const currentLoops = state.loopCount;
        return {
            messages: [response],
            loopCount: currentLoops + 1,
        };
    }

    /**
     * Node: Manually loops through tool calls requested by the LLM.
     * @param state The input graph state.
     * @returns The resulting graph state partial update.
     */
    private async executeToolsNode(state: WorkTrackerMCPAgentState): Promise<Partial<WorkTrackerMCPAgentState>> {
        const lastMessage = state.messages[state.messages.length - 1];
        if (!(lastMessage instanceof AIMessageChunk && lastMessage.tool_calls && lastMessage.tool_calls.length > 0)) {
            return {};
        }

        // Programmatic Guardrail for max. parallelly queried items
        const lastToolCallsCount = lastMessage.tool_calls.length;
        if (lastToolCallsCount > WorkTrackerMCPAgent.MAX_SIMULTANEOUS_TOOL_CALLS) {
            const errorMessages: ToolMessage[] = [];
            for (const toolCall of lastMessage.tool_calls) {
                errorMessages.push(
                    new ToolMessage(
                        `Error: Maximum of ${WorkTrackerMCPAgent.MAX_SIMULTANEOUS_TOOL_CALLS} items allowed. You queried ${lastToolCallsCount}. Please self-correct by prioritizing the most important tool calls, combining queries if possible, or executing them in sequential batches instead. Do not ask the user for help.`,
                        toolCall.id!,
                    )
                );
            }
            return { messages: errorMessages };
        }

        // Setup lists to hold our async tasks and their corresponding IDs
        const tasks: Promise<any>[] = [];
        const toolCallIds: string[] = [];

        for (const toolCall of lastMessage.tool_calls) {
            const toolFunc = this.toolsRegistry[toolCall.name];
            if (toolFunc) {
                tasks.push(toolFunc.invoke(toolCall.args));
                toolCallIds.push(toolCall.id!);
            }
        }

        // Execute all tools simultaneously, waiting for all to finish
        // Promise.allSettled prevents one failing tool from crashing the whole batch
        const results = await Promise.allSettled(tasks);

        const toolResponses = results.map((result, index) => {
            const callId = toolCallIds[index];
            let content: string;

            if (result.status === "rejected") {
                const errorMessage = result.reason instanceof Error
                    ? result.reason.message
                    : String(result.reason);
                content = `Tool execution failed: ${errorMessage}`;
            }
            else {
                // LangChain tools generally return strings, but we stringify just in case
                content = typeof result.value === "string"
                    ? result.value
                    : JSON.stringify(result.value);
            }

            return new ToolMessage(content, callId);
        });

        return { messages: toolResponses };
    }

    /**
     * Conditional edge router: Custom router acting as a programmatic circuit breaker.
     * @param state The input graph state.
     * @returns The name of the node to be routed to, or the END.
     */
    private routeOrBreak(state: WorkTrackerMCPAgentState): string {
        // Circuit Breaker Triggered: Force exit if agent loops too many times
        if (state.loopCount > WorkTrackerMCPAgent.MAX_TOOL_CALLS_ROUNDS) {
            console.log(`⚠️ Circuit breaker triggered! Agent exceeded ${WorkTrackerMCPAgent.MAX_TOOL_CALLS_ROUNDS} loops.`);
            return END;
        }

        // Standard operational routing path
        const lastMessage = state.messages[state.messages.length - 1];
        if (lastMessage instanceof AIMessageChunk && lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
            return WorkTrackerMCPAgent.TOOLS_NODE_NAME;
        }

        return END;
    }

    // --- The graph ---

    /**
     * Builds the underlying LangGraph agent graph.
     */
    private buildGraph() {
        const builder = new StateGraph(WorkTrackerMCPAgentStateAnnotation)
            // Nodes
            .addNode(WorkTrackerMCPAgent.AGENT_NODE_NAME, this.agentNode.bind(this))
            .addNode(WorkTrackerMCPAgent.TOOLS_NODE_NAME, this.executeToolsNode.bind(this))

            // Edges - a chain with the tool call loop
            .addEdge(START, WorkTrackerMCPAgent.AGENT_NODE_NAME)
            .addConditionalEdges(WorkTrackerMCPAgent.AGENT_NODE_NAME, this.routeOrBreak.bind(this), {
                [WorkTrackerMCPAgent.TOOLS_NODE_NAME]: WorkTrackerMCPAgent.TOOLS_NODE_NAME,
                [END]: END,
            })
            .addEdge(WorkTrackerMCPAgent.TOOLS_NODE_NAME, WorkTrackerMCPAgent.AGENT_NODE_NAME);

        // Compile the graph
        return builder.compile({ checkpointer: this.memoryCheckpointer });
    }
}
