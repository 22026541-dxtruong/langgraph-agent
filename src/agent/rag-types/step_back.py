from typing import Dict
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from agent.retrieval.retriever import retriever
from agent.llm import llm
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

examples = [
    {"input": "Could the members of The Police perform lawful arrests?",
     "output": "What are the powers and duties of the band The Police?"},
    {"input": "Jan Sindel's was born in what country?",
     "output": "What is Jan Sindel's personal history?"}
]

example_prompt = ChatPromptTemplate.from_messages([
    HumanMessage(content="{input}"),
    AIMessage(content="{output}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

prompt_step_back = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an expert at world knowledge. Your task is to step back and paraphrase a question "
               "to a more generic step-back question, easier to answer. Here are a few examples:"),
    few_shot_prompt,
    HumanMessage(content="{question}"),
])

generate_queries_step_back = prompt_step_back | llm | StrOutputParser()


# --------------------------
# Prompt: final response using both normal and step-back contexts
# --------------------------
response_prompt_template = """You are an expert of world knowledge. Answer the original question using 
the following contexts. Ignore irrelevant context.

# Normal Context
{normal_context}

# Step-Back Context
{step_back_context}

# Original Question: {question}
# Answer:"""

response_prompt = ChatPromptTemplate.from_template(response_prompt_template)


# --------------------------
# StepBack chain: retrieve normal + step-back context → generate answer
# --------------------------
step_back_chain = (
    {
        # Retrieve context for original question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context for step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Keep original question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | llm
    | StrOutputParser()
)

@tool
def step_back(input: str) -> str:
    """
    A RAG chain using Step-Back Prompting:
    - Generate a more general step-back version of the user query
    - Retrieve documents for both the original and step-back queries
    - Generate a final answer using combined context
    """
    return step_back_chain.invoke({"question": input})
