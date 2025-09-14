from typing import List, Tuple, Dict, Any
from json import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from agent.retrieval.retriever import retriever
from agent.llm import llm 

def reciprocal_rank_fusion(results: List[List[Dict[str, Any]]], k: int = 60) -> List[Tuple[Dict[str, Any], float]]:
    """
    Reciprocal Rank Fusion (RRF) combines multiple ranked lists of retrieved documents.
    
    Args:
        results: A list of ranked lists of documents. Each inner list is sorted by rank.
        k: Smoothing constant that controls how much weight is given to lower-ranked docs.
    
    Returns:
        A list of (document, fused_score) tuples, sorted by descending score.
    """
    fused_scores: Dict[str, float] = {}

    # Iterate through each list of ranked documents
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc, sort_keys=True)  # ensure deterministic key
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0.0
            # The RRF core: higher-ranked docs get larger score
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort documents by their new fused scores in descending order
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

template = """You are a helpful assistant that generates multiple diverse search queries 
based on a single user input.

Input query: {question}

Generate 4 related search queries (one per line).
"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_fusion
    | llm
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
)

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

@tool
def rag_fusion(input: str) -> List[Tuple[Dict[str, Any], float]]:
    """
    A RAG chain that uses Reciprocal Rank Fusion (RRF) to combine results from multiple queries.
    
    Args:
        input: The original user question.
    
    Returns:
        A list of (document, fused_score) tuples, ranked by relevance.
    """
    return retrieval_chain_rag_fusion.invoke({"question": input})
