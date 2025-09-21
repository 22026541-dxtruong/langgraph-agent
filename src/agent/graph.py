"""Main RAG workflow graph implementation."""
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from agent.state import RAGState
from agent.nodes import RAGNodes
from agent.config import Configuration


from langgraph.graph.state import CompiledStateGraph

def create_rag_graph(config: Configuration = None) -> CompiledStateGraph:
    """Create the complete RAG workflow graph."""
    
    if config is None:
        config = Configuration()
    
    if not config.is_valid:
        raise ValueError("Invalid configuration. Please check your API keys.")
    
    # Initialize nodes
    nodes = RAGNodes(config)
    
    # Create the graph
    graph = StateGraph(RAGState)
    
    # Add nodes to the graph
    graph.add_node("detect_and_load_urls", nodes.detect_and_load_urls_node)  # New node
    graph.add_node("transform_query", nodes.transform_query_node)
    graph.add_node("route_query", nodes.route_query_node)
    graph.add_node("retrieve_documents", nodes.retrieve_documents_node)
    graph.add_node("rerank_documents", nodes.rerank_documents_node)
    graph.add_node("generate_answer", nodes.generate_answer_node)
    graph.add_node("evaluate_answer", nodes.evaluate_answer_node)
    graph.add_node("improve_answer", nodes.improve_answer_node)
    
    # Define the workflow edges - START with URL detection
    graph.add_edge(START, "detect_and_load_urls")
    graph.add_edge("detect_and_load_urls", "transform_query")
    graph.add_edge("transform_query", "route_query")
    
    # Conditional routing based on route decision
    def route_decision(state: RAGState) -> str:
        """Decide next step based on routing."""
        route = state.get("route", "vectorstore")
        
        if route in ["direct_response", "websearch"]:
            return "generate_answer"
        else:
            return "retrieve_documents"
    
    graph.add_conditional_edges(
        "route_query",
        route_decision,
        {
            "retrieve_documents": "retrieve_documents",
            "generate_answer": "generate_answer"
        }
    )
    
    # Continue with retrieval flow
    graph.add_edge("retrieve_documents", "rerank_documents")
    graph.add_edge("rerank_documents", "generate_answer")
    graph.add_edge("generate_answer", "evaluate_answer")
    
    # Conditional improvement based on quality evaluation
    def should_improve(state: RAGState) -> str:
        """Decide if answer needs improvement."""
        needs_retry = state.get("needs_retry", False)
        return "improve_answer" if needs_retry else END
    
    graph.add_conditional_edges(
        "evaluate_answer",
        should_improve,
        {
            "improve_answer": "improve_answer",
            END: END
        }
    )
    
    # After improvement, re-evaluate
    graph.add_edge("improve_answer", "evaluate_answer")
    
    # Compile the graph
    return graph.compile()


def run_rag_query(question: str, config: Configuration = None) -> Dict[str, Any]:
    """Run a single RAG query and return the result."""
    
    graph = create_rag_graph(config)
    
    # Initial state
    initial_state: RAGState = {
        "messages": question,
        "transformed_queries": [],
        "route": "",
        "documents": [],
        "ranked_documents": [],
        "answer": "",
        "quality_score": 0.0,
        "needs_retry": False,
        "error": None,
        "metadata": {"step": "initialized"}
    }
    
    # Run the workflow
    try:
        result = graph.invoke(initial_state)
        return {
            "success": True,
            "answer": result.get("answer", ""),
            "quality_score": result.get("quality_score", 0.0),
            "metadata": result.get("metadata", {}),
            "error": result.get("error"),
            "loaded_urls": result.get("metadata", {}).get("loaded_urls", [])
        }
    except Exception as e:
        return {
            "success": False,
            "answer": f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}",
            "quality_score": 0.0,
            "metadata": {"error": str(e)},
            "error": str(e),
            "loaded_urls": []
        }


# Default export for LangGraph Server
graph = create_rag_graph()