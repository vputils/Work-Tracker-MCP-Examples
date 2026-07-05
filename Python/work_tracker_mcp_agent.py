from datetime import datetime
import uuid
import textwrap

import asyncio

from typing import Annotated, Any, Sequence, Optional, Callable, Self
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph, Runnable
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage, ToolMessage
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

from langchain_mcp_adapters.client import MultiServerMCPClient

from llm_providers import LLMProvider, init_model_provider
from agent_tools import get_all_local_agent_tools


DEFAULT_CHAT_MODEL_TEMPERATURE = 0.


# --- The state class ---
class _WorkTrackerMCPAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    loop_count: int


# --- The Agent class ---
class WorkTrackerMCPAgent:
    llm_provider: LLMProvider
    mcp_server_url: str
    chat_model: str
    chat_model_temperature: float

    _llm: BaseChatModel
    _llm_with_tools: Runnable[LanguageModelInput, AIMessage] | Any
    _graph: CompiledStateGraph[_WorkTrackerMCPAgentState, None, _WorkTrackerMCPAgentState, _WorkTrackerMCPAgentState]
    _memory_checkpointer: InMemorySaver

    # Initialization offloaded from the `__init__()` to `_init_mcp_and_graph()` called by the `create()` factory
    _mcp_client: MultiServerMCPClient

    _discovered_tools: list[BaseTool]
    _local_tools: list[BaseTool]  # The @tool decorator converts Callables into BaseTools
    _all_tools: list[BaseTool]
    _tools_registry: dict[str, BaseTool]

    _MAX_SIMULTANEOUS_TOOL_CALLS = 5
    _MAX_TOOL_CALLS_ROUNDS = 10  # Some of the user queries may involve many-round tool calls.

    def __init__(
        self, 
        llm_provider: LLMProvider, 
        chat_model: str,
        mcp_server_url: str,
        chat_model_temperature: float = DEFAULT_CHAT_MODEL_TEMPERATURE,
    ) -> None:
        """
        The constructor of the WorkTrackerMCPAgent class.
        NOT to be called directly, but only inside the `create()` factory method!
        """

        self.chat_model_temperature = chat_model_temperature
        self.chat_model = chat_model
        self.llm_provider = llm_provider
        self.mcp_server_url = mcp_server_url

        self._llm = init_model_provider(self.llm_provider, self.chat_model, self.chat_model_temperature)
        
        self._memory_checkpointer = InMemorySaver()

    @classmethod
    async def create(
        cls, 
        llm_provider: LLMProvider, 
        chat_model: str, 
        mcp_server_url: str,
        chat_model_temperature: float = DEFAULT_CHAT_MODEL_TEMPERATURE
    ) -> Self:
        """The async factory for the WorkTrackerMCPAgent."""
        instance = cls(llm_provider, chat_model, mcp_server_url, chat_model_temperature)
        await instance._init_mcp_and_graph()
        return instance

    async def run_single_query(self, user_message: str, thread_id: Optional[str] = None) -> str:
        """
        Process a single user query.

        Returns: The resulting Thread ID, which can be later used to resume the in-memory saved chat thread.
        """

        if thread_id is None:
            thread_id = str(uuid.uuid4())
        config = {"thread_id": thread_id}

        initial_state: _WorkTrackerMCPAgentState = {
            "messages": [HumanMessage(user_message)],
            "loop_count": 0
        }
        
        result = await self._graph.ainvoke(initial_state, config=RunnableConfig(configurable=config))

        print("\nAssistant: " + result["messages"][-1].content)

        return thread_id

    async def run_streaming_loop(self, cancellation_callback: Callable[[], bool], thread_id: Optional[str] = None) -> str:
        """
        Run a CLI streaming user chat loop.
        Exit from the loop by pressing Ctrl+C during the agent's thinking or answering,
        or by pressing Ctrl+D when waiting for a user input.

        Returns: The resulting Thread ID, which can be later used to resume the in-memory saved chat thread.
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        config = {"thread_id": thread_id}

        print("--- Session Started ---\n")
        print("In order to terminate the agent,"
              "\npress Ctrl+C when the agent is thinking or answering"
              "\nor Ctrl+D when waiting for the user input.\n")

        while not cancellation_callback():
            try:
                user_message = await asyncio.to_thread(input, "User:      ")
            except EOFError:
                break

            passed_state: _WorkTrackerMCPAgentState = {
                "messages": [HumanMessage(user_message)],
                "loop_count": 0
            }

            print("\nThinking...", end="\r")  # So that it can be rewritten

            prefix_printed = False
            async for part in self._graph.astream(
                    passed_state, config=RunnableConfig(configurable=config), stream_mode="messages", version="v2"):
                if part["type"] != "messages":
                    if cancellation_callback():
                        break
                    continue
                   
                msg, metadata = part["data"]

                # We target content chunks originating specifically from our 'agent' node
                if metadata.get("langgraph_node") == self._AGENT_NODE_NAME:
                    # Check if the chunk contains text content (ignores empty tool-call chunks)
                    if msg.content:
                        if not prefix_printed:
                            print("Assistant: ", end="", flush=True)
                            prefix_printed = True

                        print(msg.content, end="", flush=True)

                if cancellation_callback():
                    break
            
            if prefix_printed:
                print("\n\n", end="")

        return thread_id

    
    # --- MCP Discovery and Graph initialization ---
    async def _init_mcp_and_graph(self):
        """Asynchronously connects to the Work Tracker MCP server, discovers tools, and builds the graph."""
        print(f"Connecting to Work Tracker MCP server at '{self.mcp_server_url}'...")
        
        # Connect to the MCP server
        self._mcp_client = MultiServerMCPClient({
            "worktracker": {
                "transport": "streamable_http",
                "url": self.mcp_server_url
            }
        })
        
        # Automatically fetch all tools exposed by the MCP server (list_spaces, start_work_day, etc.)
        self._discovered_tools = await self._mcp_client.get_tools()

        # Combine local tools and remote MCP tools
        self._local_tools = get_all_local_agent_tools()
        self._all_tools = self._discovered_tools + self._local_tools
        
        # Build the tools registry and and bind them to the LLM
        self._tools_registry = {tool.name: tool for tool in self._all_tools}
        self._llm_with_tools = self._llm.bind_tools(list(self._tools_registry.values()))
        
        # Now that tools are bound, compile the graph
        self._graph = self._build_graph()
        
        print(f"Successfully connected! Discovered {len(self._discovered_tools)} MCP tools and adding {len(self._local_tools)} local tools.")


    # --- The nodes ---
    _AGENT_NODE_NAME = "agent"
    _TOOLS_NODE_NAME = "tools"

    async def _agent_node(self, state: _WorkTrackerMCPAgentState) -> dict[str, Any]:
        """Node: Invokes the LLM with the current conversation history, prepending the system message."""
        local_tools_names = [_tool.name for _tool in self._local_tools]
        local_tools_addend = f' along with a set of local tools (`{"`, `".join(local_tools_names)}`)' if len(local_tools_names) > 0 else ''

        system_message = textwrap.dedent(f"""
            You are a highly capable, integrated executive assistant connected directly to the user's Work Tracker application via the Model Context Protocol (MCP).
            Your primary role is to seamlessly manage their workspaces, tasks, work days, and time logging.

            ### Operational Guidelines:
            1. **Auto-Discovery:** 
                You have access to a suite of Work Tracker tools (e.g., `list_actionable_tasks`, `start_work_day`, `start_task_work`){local_tools_addend}. 
                Use them proactively to fulfill the user's intent.
            2. **State Dependencies:** 
                If the user asks to "start working on a task", you MUST check if there is an active work day using `get_current_work_status`.
                If there isn't one, invoke `start_work_day` before invoking `start_task_work`.
            3. **Data Retrieval:**
                If asked for a summary, utilize tools like `get_time_report` or `list_work_days` to provide accurate, up-to-date metrics. 
            4. **Destructive Actions:** 
                Proceed carefully with deletions (`delete_task`, `delete_work_day`). 
                If the user request is ambiguous, ask for confirmation before executing.
            5. **Efficiency:** 
                You may execute up to {self._MAX_SIMULTANEOUS_TOOL_CALLS} tool calls simultaneously and before giving an answer to the user, in only at most {self._MAX_TOOL_CALLS_ROUNDS} rounds. 
                Therefore if you were about to call tools in more than {self._MAX_TOOL_CALLS_ROUNDS}, answer to the user after at most {self._MAX_TOOL_CALLS_ROUNDS} tool call rounds and ask the user for a follow-up if they want to proceed.
                Do not invent task IDs; search for them if the user provides a name instead of an ID using `search_tasks`.

            Communicate concisely and professionally. You are assisting a power user who values speed and exactness.

            The current date and time is: {datetime.now().strftime('%A, %Y-%m-%d %H:%M:%S')}.
        """).strip()

        all_messages = [SystemMessage(system_message), *state["messages"]]

        # The LLM looks at the history and decides to either reply text or output a tool call.
        response = await self._llm_with_tools.ainvoke(all_messages)
        current_loops = state.get("loop_count", 0)
        return {"messages": [response], "loop_count": current_loops + 1}
        
    async def _execute_tools_node(self, state: _WorkTrackerMCPAgentState) -> dict[str, Any]:
        """Node: Manually loops through tool calls requested by the LLM."""
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return {}

        last_tool_calls_count = len(last_message.tool_calls)
        if last_tool_calls_count == 0:
            return {}  # Early exit

        # Programmatic Guardrail for max. parallelly queried items
        if last_tool_calls_count > self._MAX_SIMULTANEOUS_TOOL_CALLS:
            error_messages = []
            for tool_call in last_message.tool_calls:
                error_messages.append(
                    ToolMessage(
                        content=f"Error: Maximum of {self._MAX_SIMULTANEOUS_TOOL_CALLS} items allowed. You queried {last_tool_calls_count}. Please self-correct by prioritizing the most important tool calls, combining queries if possible, or executing them in sequential batches instead. Do not ask the user for help.", 
                        tool_call_id=tool_call["id"]
                    )
                )
            return {"messages": error_messages}
        
        # Setup lists to hold our async tasks and their corresponding IDs
        tasks = []
        tool_call_ids = []

        for tool_call in last_message.tool_calls:
            tool_func = self._tools_registry.get(tool_call["name"])
            if tool_func:
                # We call .ainvoke() universally. LangChain handles the sync-to-thread translation.
                tasks.append(tool_func.ainvoke(tool_call["args"]))
                tool_call_ids.append(tool_call["id"])
                
        # Execute all tools simultaneously, waiting for all to finish
        # return_exceptions=True prevents one failing tool from crashing the whole batch
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Map the results back to ToolMessages
        tool_responses = []
        for result, call_id in zip(results, tool_call_ids):
            if isinstance(result, Exception):
                content = f"Tool execution failed: {str(result)}"
            else:
                content = str(result)
                
            tool_responses.append(ToolMessage(content=content, tool_call_id=call_id))
                
        return {"messages": tool_responses}
        
    def _route_or_break(self, state: _WorkTrackerMCPAgentState) -> str:
        """Conditional edge router: Custom router acting as a programmatic circuit breaker."""        
        # Circuit Breaker Triggered: Force exit if agent loops too many times
        if state.get("loop_count", 0) > self._MAX_TOOL_CALLS_ROUNDS:
            print(f"⚠️ Circuit breaker triggered! Agent exceeded {self._MAX_TOOL_CALLS_ROUNDS} loops.")
            return END
            
        # Standard operational routing path
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return self._TOOLS_NODE_NAME
            
        return END


    # --- The graph ---
    def _build_graph(self):
        """Builds the underlying LangGraph agent graph."""
        builder = StateGraph(_WorkTrackerMCPAgentState)

        # Nodes
        builder.add_node(self._AGENT_NODE_NAME, self._agent_node)
        builder.add_node(self._TOOLS_NODE_NAME, self._execute_tools_node)

        # Edges - a chain with the tool call loop
        builder.add_edge(START, self._AGENT_NODE_NAME)
        builder.add_conditional_edges(self._AGENT_NODE_NAME, self._route_or_break, {
            # Identity dictionary needed just to avoid reflection runtime errors
            self._TOOLS_NODE_NAME: self._TOOLS_NODE_NAME,
            END: END
        })
        builder.add_edge(self._TOOLS_NODE_NAME, self._AGENT_NODE_NAME)

        # Compile the graph
        return builder.compile(checkpointer=self._memory_checkpointer)
