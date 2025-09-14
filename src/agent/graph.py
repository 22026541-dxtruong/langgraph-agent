from langgraph.graph import StateGraph
from agent.rag_types.query_optimizer import query_optimizer_node
from typing import TypedDict
from agent.state import GraphState

graph = StateGraph(GraphState)
graph.add_node("query_optimizer", query_optimizer_node)
graph.set_entry_point("query_optimizer")
graph = graph.compile()