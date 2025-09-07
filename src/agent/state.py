from typing import Annotated, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """The state of our agent."""
    
    # Messages between human and agent
    messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list,
        description="Messages in the conversation"
    )
    
    # Research and retrieval results
    research_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results from web research"
    )
    
    retrieval_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results from document retrieval"
    )
    
    # Memory and context - Enhanced
    conversation_summary: Optional[str] = Field(
        default=None,
        description="Summary of previous conversation"
    )
    
    user_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="User context and preferences"
    )
    
    memory_categories: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of items in each memory category"
    )
    
    active_reminders: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active reminders for the user"
    )
    
    # Agent reasoning
    thoughts: List[str] = Field(
        default_factory=list,
        description="Agent's internal thoughts and reasoning"
    )
    
    current_task: Optional[str] = Field(
        default=None,
        description="Current task being worked on"
    )
    
    # Tools and actions
    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of tool calls made"
    )
    
    next_action: Optional[str] = Field(
        default=None,
        description="Next action to take"
    )