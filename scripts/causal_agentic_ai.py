"""
Causal AI Agentic Framework - Main Module

This is the primary entry point for the causal AI agentic framework.
It imports all utilities and tools from causal_forest_optimizer and tools modules,
and provides agent building functionality.

Dependencies:
- causal_forest_optimizer.py: Standalone causal forest utilities
- tools.py: Core tools and helpers (imports from causal_forest_optimizer)
"""

from typing import Annotated, List, Literal, Optional, TypedDict, Any, Dict

# Import all from tools module
from tools import (
    # State management
    GLOBAL_DATA,
    SESSION_STATE,
    init_llm,
    set_global_data,
    set_last_user_question,
    # Config
    COMPREHENSIVE_MODEL_PATH,
    # Helpers
    load_comprehensive_model,
    normalize_store_dept_keys,
    determine_data_subsets,
    map_variable_name_intelligently,
    parse_date_intelligently,
    # Tools
    analyze_whatif_scenario,
    find_optimal_levers,
    analyze_variable_impact,
)

# Optional imports for agent building
try:
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, END, START
    from langgraph.prebuilt import ToolNode
    from langgraph.graph.message import add_messages
except Exception:  # pragma: no cover
    ChatOpenAI = None
    StateGraph = None
    ToolNode = None
    add_messages = lambda xs: xs
    START = "START"
    END = "END"


# ============================================================================
# AGENT BUILDING (Main Export)
# ============================================================================

class AgentState(TypedDict):
    """State container for LangGraph agent."""
    messages: Annotated[List, add_messages]
    iterations: int
    final_response: str


def build_agent(tools: Optional[List] = None, model: str = "gpt-4o-mini", temperature: float = 0) -> tuple:
    """Build and return a LangGraph agent with the causal analysis tools.

    Args:
        tools: List of tool functions. Defaults to [find_optimal_levers, analyze_whatif_scenario, analyze_variable_impact]
        model: LLM model name (default: "gpt-4o-mini")
        temperature: LLM temperature (default: 0)

    Returns:
        (agent_executor, llm, llm_with_tools): Compiled agent and LLM instances

    Raises:
        RuntimeError: If langchain_openai or langgraph is not installed
    """
    if ChatOpenAI is None or StateGraph is None or ToolNode is None:
        raise RuntimeError(
            "Agent stack not available. Install with: "
            "pip install langchain_openai langgraph"
        )

    # Initialize LLM
    llm = init_llm(model=model, temperature=temperature)
    if llm is None:
        llm = ChatOpenAI(model=model, temperature=temperature)

    # Default tools
    tools = tools or [find_optimal_levers, analyze_whatif_scenario, analyze_variable_impact]
    llm_with_tools = llm.bind_tools(tools)

    # Define agent nodes
    def call_model(state: AgentState) -> AgentState:
        """Agent node: Call the LLM to decide next action"""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {
            "messages": [response],
            "iterations": state.get("iterations", 0) + 1
        }

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Routing function: Determine if we should use tools or end"""
        messages = state["messages"]
        last_message = messages[-1]

        # Check if max iterations reached
        if state.get("iterations", 0) > 10:
            return "end"

        # If LLM makes a tool call, continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # Otherwise, end
        return "end"

    # Build workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    agent_executor = workflow.compile()
    return agent_executor, llm, llm_with_tools


# ============================================================================
# PUBLIC API (Re-exports)
# ============================================================================

__all__ = [
    # State management
    "GLOBAL_DATA",
    "SESSION_STATE",
    "init_llm",
    "set_global_data",
    "set_last_user_question",
    # Config
    "COMPREHENSIVE_MODEL_PATH",
    # Helpers
    "load_comprehensive_model",
    "normalize_store_dept_keys",
    "determine_data_subsets",
    "map_variable_name_intelligently",
    "parse_date_intelligently",
    # Tools
    "analyze_whatif_scenario",
    "find_optimal_levers",
    "analyze_variable_impact",
    # Agent
    "AgentState",
    "build_agent",
]
