from typing import Dict, List
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from agent.llm import llm

from agent.rag_types.multi_query import multi_query
from agent.rag_types.rag_fusion import rag_fusion
from agent.rag_types.decomposition import decomposition
from agent.rag_types.step_back import step_back
from agent.rag_types.hyde import hyde
from agent.graph import GraphState

PROMPT_QUERY_OPTIMIZER = """You are a Query Optimization Agent for a RAG system.
Given a user query, select the most suitable strategy to optimize retrieval.

Available strategies:
1. MultiQuery → Generate multiple semantic variants of the query.
2. RAGFusion → Generate multiple related queries and combine results.
3. Decomposition → Break complex question into simpler sub-questions.
4. StepBack → Rewrite query into a more general question.
5. HyDE → Generate a hypothetical document to guide retrieval.
"""

# --------------------------
# Router Node
# --------------------------
tools = [multi_query, decomposition, hyde, rag_fusion, step_back]
def query_optimizer_node(state: GraphState) -> GraphState:
    """
    Router node that selects the appropriate tool and returns results directly.

    Input: {"query": str}
    Output: {"strategy": str}
    """
    user_query = state["query"]

    # 1️⃣ Ask LLM which strategy to use
    prompt = f"{PROMPT_QUERY_OPTIMIZER}\nUser query: \"{user_query}\""
    resp = llm.invoke(prompt)
    return {"result": resp.content, "query": user_query}
