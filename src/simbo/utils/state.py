"""
State definitions for Simbo agents using LangGraph.
"""

from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class ConversationMessage(BaseModel):
    """Represents a single message in the conversation."""
    role: str = Field(..., description="Role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class WorkspaceInfo(BaseModel):
    """Information about the ROS workspace."""
    path: str = Field(default="", description="Path to ROS workspace")
    ros_version: str = Field(default="", description="ROS version (ros1 or ros2)")
    ros_distro: str = Field(default="", description="ROS distribution name")
    gazebo_version: str = Field(default="", description="Gazebo version")
    packages: List[str] = Field(default_factory=list, description="List of packages")
    source_files: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Source files per package"
    )
    launch_files: List[str] = Field(default_factory=list, description="Launch files")
    is_analyzed: bool = Field(default=False, description="Whether workspace has been analyzed")


class AgentState(TypedDict):
    """State for the Simbo simulation agent."""
    # Conversation messages with reducer for accumulation
    messages: Annotated[Sequence[Any], add_messages]

    # Workspace information
    workspace: Optional[WorkspaceInfo]

    # Current stage in the workflow
    stage: str  # "init", "workspace_setup", "analysis", "coding", "feedback"

    # User's current request/intent
    current_intent: str

    # Generated code waiting for feedback
    pending_code: Optional[Dict[str, str]]

    # Feedback history
    feedback_history: List[str]

    # Error messages if any
    error: Optional[str]

    # Whether agent needs user input
    needs_user_input: bool

    # Question to ask user
    user_question: Optional[str]


class CodeGenerationResult(BaseModel):
    """Result of code generation."""
    file_path: str = Field(..., description="Path where code should be written")
    code: str = Field(..., description="Generated code content")
    language: str = Field(default="python", description="Programming language")
    description: str = Field(default="", description="Description of what the code does")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")


class AnalysisResult(BaseModel):
    """Result of workspace/code analysis."""
    summary: str = Field(..., description="Summary of the analysis")
    findings: List[str] = Field(default_factory=list, description="Key findings")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    relevant_files: List[str] = Field(default_factory=list, description="Relevant files found")
