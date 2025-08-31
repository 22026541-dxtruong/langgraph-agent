import os
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document


# Initialize DuckDuckGo search
search_tool = TavilySearchResults(max_results=4)

# Initialize embeddings for retrieval
import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", async_client=False)

# In-memory document store (in production, use persistent storage)
document_store = None


@tool
def web_search(query: str) -> str:
    """
    Search the web for information using DuckDuckGo.
    
    Args:
        query: Search query
    
    Returns:
        Search results as formatted string
    """
    try:
        results = search_tool.invoke(query)
        return results
    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool
def add_documents_to_retrieval(documents: List[str]) -> str:
    """
    Add documents to the retrieval system.
    
    Args:
        documents: List of document texts to add
    
    Returns:
        Status message
    """
    global document_store
    
    try:
        docs = [Document(page_content=doc) for doc in documents]
        
        if document_store is None:
            document_store = FAISS.from_documents(docs, embeddings)
        else:
            document_store.add_documents(docs)
        
        return f"Successfully added {len(documents)} documents to retrieval system"
    except Exception as e:
        return f"Failed to add documents: {str(e)}"


@tool
def retrieve_documents(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents from the knowledge base.
    
    Args:
        query: Search query
        k: Number of documents to retrieve
    
    Returns:
        List of relevant documents with content and metadata
    """
    global document_store
    
    if document_store is None:
        return [{"error": "No documents in retrieval system"}]
    
    try:
        docs = document_store.similarity_search(query, k=k)
        
        results = []
        for i, doc in enumerate(docs):
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "rank": i + 1
            })
        
        return results
    except Exception as e:
        return [{"error": f"Retrieval failed: {str(e)}"}]


@tool
def save_to_memory(key: str, value: str) -> str:
    """
    Save information to long-term memory.
    
    Args:
        key: Memory key
        value: Value to store
    
    Returns:
        Status message
    """
    # In production, use persistent storage like Redis or database
    if not hasattr(save_to_memory, "memory"):
        save_to_memory.memory = {}
    
    save_to_memory.memory[key] = value
    return f"Saved to memory: {key}"


@tool
def recall_from_memory(key: str) -> str:
    """
    Recall information from long-term memory.
    
    Args:
        key: Memory key to retrieve
    
    Returns:
        Stored value or error message
    """
    if not hasattr(save_to_memory, "memory"):
        return f"No memory found for key: {key}"
    
    value = save_to_memory.memory.get(key)
    if value is None:
        return f"No memory found for key: {key}"
    
    return value


@tool
def calculate(expression: str) -> str:
    """
    Safely calculate mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate
    
    Returns:
        Result of calculation or error message
    """
    try:
        # Simple and safe calculation
        allowed_chars = "0123456789+-*/()."
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


# Export all tools
all_tools = [
    web_search,
    add_documents_to_retrieval,
    retrieve_documents,
    save_to_memory,
    recall_from_memory,
    calculate
]