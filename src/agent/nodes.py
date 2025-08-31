from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.tools import all_tools


# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# Create tool node for executing tools
tool_node = ToolNode(all_tools)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(all_tools)


# System prompt for the ReAct agent
# SYSTEM_PROMPT = """You are a helpful AI assistant with access to various tools including web search, webpage content fetching, document retrieval, memory, and calculation capabilities.

# You follow a ReAct (Reasoning and Acting) approach:
# 1. **Think** about the user's request and what you need to do
# 2. **Act** by using appropriate tools if needed
# 3. **Observe** the results from tools
# 4. **Respond** with a helpful answer based on the tool results

# Available capabilities:
# - **Research**: Use web_search to find search results
# - **Retrieval**: Use retrieve_documents to search your knowledge base
# - **Memory**: Use save_to_memory and recall_from_memory for long-term storage
# - **Calculation**: Use calculate for mathematical operations
# - **Document Management**: Use add_documents_to_retrieval to add new documents

# Guidelines:
# - Always USE TOOLS when you need current information or external data
# - Don't just plan - actually execute the tools to get real results
# - Wait for tool results before responding to the user
# - Synthesize information from tool results in your final response

# IMPORTANT: When user asks about current information (weather, news, etc.), you MUST use tools to get real data. Don't just acknowledge the need to search - actually do it!

# Current conversation summary: {conversation_summary}
# User context: {user_context}
# """
SYSTEM_PROMPT = """You are a helpful AI assistant with access to various tools including web search, document retrieval, memory, and calculation capabilities.

You follow a ReAct (Reasoning and Acting) approach:
1. **Think** about the user's request and what you need to do
2. **Act** by using appropriate tools if needed
3. **Observe** the results
4. **Respond** with a helpful answer

Available capabilities:
- **Research**: Use web_search to find current information online
- **Retrieval**: Use retrieve_documents to search your knowledge base
- **Memory**: Use save_to_memory and recall_from_memory for long-term storage
- **Calculation**: Use calculate for mathematical operations
- **Document Management**: Use add_documents_to_retrieval to add new documents

Guidelines:
- Always think through your approach before acting
- Use web_search to find current information online
- Use tools when you need external information or capabilities
- Summarize and synthesize information from multiple sources
- Maintain context and memory of important information
- Be thorough but concise in your responses

Current conversation summary: {conversation_summary}
User context: {user_context}
"""

async def agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Main agent reasoning and decision-making node.
    """
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    # Format the prompt with current state
    formatted_prompt = prompt.format_messages(
        conversation_summary=state.conversation_summary or "No previous conversation",
        user_context=str(state.user_context) if state.user_context else "No additional context",
        messages=state.messages
    )
    
    # Get response from LLM
    response = await llm_with_tools.ainvoke(formatted_prompt)
    
    # Add agent's reasoning to thoughts
    thoughts = state.thoughts.copy()
    thoughts.append(f"Agent response: {response.content if response.content else 'Tool calls planned'}")
    
    return {
        "messages": [response],
        "thoughts": thoughts
    }


async def memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Handle memory operations and conversation summarization.
    """
    # If conversation is getting long, create a summary
    if len(state.messages) > 10 and not state.conversation_summary:
        # Create summary prompt
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the key points from this conversation in 2-3 sentences."),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        formatted_prompt = summary_prompt.format_messages(messages=state.messages[:-2])  # Exclude last 2 messages
        summary_response = await llm.ainvoke(formatted_prompt)
        
        return {
            "conversation_summary": summary_response.content,
            "thoughts": state.thoughts + ["Updated conversation summary"]
        }
    
    return {"thoughts": state.thoughts}


async def research_node(state: AgentState) -> Dict[str, Any]:
    """
    Handle research and web search operations.
    """
    # Check if the last message contains tool calls for web search
    last_message = state.messages[-1] if state.messages else None
    
    if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == 'web_search':
                return {"thoughts": state.thoughts + ["Initiating web search"]}
    
    return {"thoughts": state.thoughts}


async def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    Handle document retrieval operations.
    """
    # Check if the last message contains tool calls for retrieval
    last_message = state.messages[-1] if state.messages else None
    
    if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call['name'] in ['retrieve_documents', 'add_documents_to_retrieval']:
                return {"thoughts": state.thoughts + ["Processing document retrieval"]}
    
    return {"thoughts": state.thoughts}


def should_continue(state: AgentState) -> str:
    """
    Determine whether to continue with tool execution or end.
    """
    last_message = state.messages[-1] if state.messages else None
    
    # Check if the last message has tool calls
    if (last_message and 
        hasattr(last_message, 'tool_calls') and 
        last_message.tool_calls and 
        len(last_message.tool_calls) > 0):
        return "tools"
    
    return "end"


def route_tools(state: AgentState) -> str:
    """
    Route to appropriate tool handling based on the tool being called.
    """
    last_message = state.messages[-1] if state.messages else None
    
    if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_name = last_message.tool_calls[0]['name']
        
        if tool_name == 'web_search':
            return "research"
        elif tool_name in ['retrieve_documents', 'add_documents_to_retrieval']:
            return "retrieval"
        elif tool_name in ['save_to_memory', 'recall_from_memory']:
            return "memory"
        else:
            return "tools"
    
    return "agent"