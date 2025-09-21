"""Graph state management for RAG system."""
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict


class RAGState(TypedDict):
    """State for the RAG workflow."""
    messages: str
    transformed_queries: List[str]
    route: str  # "vectorstore", "websearch", "direct_response"
    # Retrieved documents
    documents: List[Dict[str, Any]]
    ranked_documents: List[Dict[str, Any]]
    answer: str
    quality_score: float
    needs_retry: bool
    error: Optional[str]
    metadata: Dict[str, Any]
