from typing import List, Dict, Any
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from agent.retrieval.retriever import retriever
from agent.llm import llm
from langchain.load import dumps, loads

template_perspectives = """You are an AI assistant. 
Your task is to generate five different versions of the given user question 
to retrieve relevant documents from a vector database. 

By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of distance-based similarity search. 

Original question: {question}

Output: 5 alternative phrasings, one per line.
"""

prompt_perspectives = ChatPromptTemplate.from_template(template_perspectives)

generate_queries = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
)

def get_unique_union(documents: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Flatten lists of retrieved documents and remove duplicates.

    Args:
        documents: List of document lists (from multiple queries).
    Returns:
        Unique list of documents.
    """
    flattened_docs = [dumps(doc, sort_keys=True) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

template_answer = """Answer the following question based on the given context.

Context:
{context}

Question: {question}
"""

prompt_answer = ChatPromptTemplate.from_template(template_answer)

final_rag_chain = (
    {"context": generate_queries | retriever.map() | get_unique_union,
     "question": itemgetter("question")}
    | prompt_answer
    | llm
    | StrOutputParser()
)

@tool
def multi_query(input: str) -> str:
    """
    A RAG chain that uses Multi-Query strategy:
    - Generate multiple diverse queries for the same question
    - Retrieve documents for each query
    - Merge results with unique union
    - Generate a final answer based on merged context
    """
    return final_rag_chain.invoke({"question": input})
