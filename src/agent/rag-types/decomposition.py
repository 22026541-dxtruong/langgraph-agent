from typing import List
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from agent.retrieval.retriever import retriever
from agent.llm import llm

template_decomposition = """You are a helpful assistant that decomposes complex questions 
into simpler, self-contained sub-questions.

Goal:
- Break the input into 3 focused sub-questions that can each be answered independently.
- Ensure sub-questions are specific, concise, and non-overlapping.

Original question: {question}

Output: 3 sub-questions, one per line.
"""

prompt_decomposition = ChatPromptTemplate.from_template(template_decomposition)

generate_subquestions = (
    prompt_decomposition
    | llm
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
)

template_rag = """You are an expert assistant. 
Use the following retrieved context to answer the question concisely.

Context:
{context}

Question: {question}

Answer:"""

prompt_rag = ChatPromptTemplate.from_template(template_rag)


def answer_subquestions(sub_questions: List[str]) -> List[str]:
    """Retrieve documents and answer each sub-question."""
    answers = []
    for sub_q in sub_questions:
        docs = retriever.get_relevant_documents(sub_q)
        context = "\n".join([doc.page_content for doc in docs])
        answer = (prompt_rag | llm | StrOutputParser()).invoke(
            {"context": context, "question": sub_q}
        )
        answers.append(answer)
    return answers


def format_qa_pairs(questions: List[str], answers: List[str]) -> str:
    """Format Q&A pairs into a context string."""
    return "\n\n".join(
        [f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(zip(questions, answers))]
    )

template_synthesis = """Here are Q&A pairs derived from a decomposition of the question:

{context}

Now, synthesize a final comprehensive answer to the original question:
{question}
"""

prompt_synthesis = ChatPromptTemplate.from_template(template_synthesis)

final_synthesis_chain = (
    prompt_synthesis
    | llm
    | StrOutputParser()
)

@tool
def decomposition(input: str) -> str:
    """
    A RAG chain that uses Decomposition strategy:
    - Breaks a complex question into 3 simpler sub-questions
    - Retrieves and answers each sub-question individually
    - Synthesizes final comprehensive answer from Q&A pairs
    """
    sub_questions = generate_subquestions.invoke({"question": input})
    answers = answer_subquestions(sub_questions)
    context = format_qa_pairs(sub_questions, answers)
    return final_synthesis_chain.invoke({"context": context, "question": input})
