from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    agent_node,
    memory_node,
    research_node,
    retrieval_node,
    tool_node,
    should_continue,
    route_tools
)


def create_graph():
    """
    Create the main agent graph with all nodes and edges.
    """
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("research", research_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Tool routing - route to specific handlers first, then back to tools
    workflow.add_conditional_edges(
        "tools",
        route_tools,
        {
            "research": "research",
            "retrieval": "retrieval",
            "memory": "memory",
            "agent": "agent",
            "tools": "agent"  # Default back to agent
        }
    )
    
    # From specialized nodes back to agent
    workflow.add_edge("research", "agent")
    workflow.add_edge("retrieval", "agent")
    workflow.add_edge("memory", "agent")
    
    # Compile the graph with memory
    # memory = MemorySaver()
    # graph = workflow.compile(checkpointer=memory)
    graph = workflow.compile()
    
    return graph


# Create the main graph instance
graph = create_graph()


async def run_agent(message: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run the agent with a message.
    
    Args:
        message: User message
        config: Configuration including thread_id for memory
    
    Returns:
        Agent response
    """
    if config is None:
        config = {"configurable": {"thread_id": "default"}}
    
    # Create initial state
    initial_state = AgentState(messages=[{"role": "user", "content": message}])
    
    # Run the graph
    final_state = await graph.ainvoke(initial_state, config=config)
    
    return {
        "response": final_state.messages[-1].content if final_state.messages else "No response",
        "thoughts": final_state.thoughts,
        "research_results": final_state.research_results,
        "retrieval_results": final_state.retrieval_results
    }
