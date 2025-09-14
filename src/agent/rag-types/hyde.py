from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from agent.retrieval.retriever import retriever
from agent.llm import llm
from langchain_core.documents import Document

# --------------------------
# Prompt: generate hypothetical document
# --------------------------
template_hyde = """Please write a detailed, scientific-style passage that answers the question.
Question: {question}
Passage:"""

prompt_hyde = ChatPromptTemplate.from_template(template_hyde)

generate_hypothetical_doc = prompt_hyde | llm | StrOutputParser()


# --------------------------
# Chain: retrieve real documents using HyDE passage
# --------------------------
def retrieve_with_hyde(question: str) -> List[Document]:
    """
    Generate a hypothetical document and use it to retrieve semantically similar real documents.
    """
    hypothetical_doc = generate_hypothetical_doc.invoke({"question": question})
    
    # Retrieve using the hypothetical document as input
    retrieved_docs = retriever.get_relevant_documents(hypothetical_doc)
    return retrieved_docs


# --------------------------
# Tool wrapper
# --------------------------
@tool
def hyde(input: str) -> str:
    """
    A RAG chain using HyDE strategy:
    - Generate a hypothetical document for the user question
    - Retrieve real documents that are semantically similar to the hypothetical doc
    - Generate a final answer from the retrieved context
    """
    retrieved_docs = retrieve_with_hyde(input)
    
    # Format context
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Standard RAG prompt for final answer
    final_prompt_template = """Answer the following question using the retrieved context. Ignore irrelevant information.

Context:
{context}

Question: {question}
Answer:"""
    final_prompt = ChatPromptTemplate.from_template(final_prompt_template)
    
    final_rag_chain = final_prompt | llm | StrOutputParser()
    
    return final_rag_chain.invoke({"context": context, "question": input})
