"""
Simulation Agent - Main LangGraph-based agent for Simbo.

This agent handles the complete workflow:
1. User prompt processing
2. Workspace detection and analysis
3. Environment learning
4. Code generation
5. Feedback loop
"""

import os
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from ..utils.state import AgentState, WorkspaceInfo
from ..tools.workspace_tools import (
    detect_ros_version,
    analyze_workspace,
    list_packages,
    read_package_xml,
    find_launch_files,
    find_source_files,
)
from ..tools.code_tools import (
    read_file,
    write_file,
    search_code,
    generate_controller,
)


# System prompt for the simulation agent
SYSTEM_PROMPT = """You are Simbo, an expert AI assistant for ROS/Gazebo robotics simulation development.

Your capabilities include:
1. Analyzing ROS workspaces (both ROS1 and ROS2)
2. Understanding existing robot simulations and code
3. Generating control programs for robots
4. Creating launch files and configuration
5. Helping debug simulation issues

Current workspace information:
{workspace_info}

Guidelines:
- Always analyze the user's workspace before making suggestions
- Generate code that follows ROS best practices
- Explain your reasoning and the code you generate
- Ask clarifying questions when requirements are unclear
- Provide complete, runnable code solutions
- Consider both ROS1 and ROS2 compatibility based on the detected environment

When generating controllers:
- Use appropriate message types for the ROS version
- Include proper error handling
- Add helpful comments and documentation
- Make parameters configurable

Current stage: {stage}
"""


class SimulationAgent:
    """Main simulation agent using LangGraph."""

    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
        api_key: Optional[str] = None
    ):
        """
        Initialize the simulation agent.

        Args:
            model_name: OpenAI model to use
            temperature: Model temperature for generation
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.api_key
        )

        # Define tools
        self.tools = [
            detect_ros_version,
            analyze_workspace,
            list_packages,
            read_package_xml,
            find_launch_files,
            find_source_files,
            read_file,
            write_file,
            search_code,
            generate_controller,
        ]

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("check_workspace", self._check_workspace)
        workflow.add_node("analyze_environment", self._analyze_environment)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("execute_tools", ToolNode(self.tools))
        workflow.add_node("handle_feedback", self._handle_feedback)
        workflow.add_node("ask_user", self._ask_user)

        # Set entry point
        workflow.set_entry_point("process_input")

        # Add edges
        workflow.add_conditional_edges(
            "process_input",
            self._route_after_input,
            {
                "check_workspace": "check_workspace",
                "generate_response": "generate_response",
                "ask_user": "ask_user",
            }
        )

        workflow.add_conditional_edges(
            "check_workspace",
            self._route_after_workspace_check,
            {
                "analyze_environment": "analyze_environment",
                "ask_user": "ask_user",
            }
        )

        workflow.add_edge("analyze_environment", "generate_response")

        workflow.add_conditional_edges(
            "generate_response",
            self._route_after_response,
            {
                "execute_tools": "execute_tools",
                "handle_feedback": "handle_feedback",
                "end": END,
            }
        )

        workflow.add_conditional_edges(
            "execute_tools",
            self._route_after_tools,
            {
                "generate_response": "generate_response",
                "end": END,
            }
        )

        workflow.add_edge("handle_feedback", "generate_response")
        workflow.add_edge("ask_user", END)

        return workflow.compile()

    def _process_input(self, state: AgentState) -> AgentState:
        """Process user input and determine intent."""
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        content = last_message.content if hasattr(last_message, 'content') else str(last_message)

        # Determine intent from message
        intent_keywords = {
            "workspace": ["workspace", "directory", "folder", "path", "setup"],
            "analyze": ["analyze", "understand", "learn", "explore", "show"],
            "generate": ["generate", "create", "write", "make", "build", "controller", "code"],
            "help": ["help", "how", "what", "explain"],
            "feedback": ["feedback", "change", "modify", "update", "fix"],
        }

        detected_intent = "general"
        content_lower = content.lower()

        for intent, keywords in intent_keywords.items():
            if any(kw in content_lower for kw in keywords):
                detected_intent = intent
                break

        return {
            **state,
            "current_intent": detected_intent,
            "error": None,
        }

    def _check_workspace(self, state: AgentState) -> AgentState:
        """Check if workspace is set and valid."""
        workspace = state.get("workspace")

        if workspace and workspace.path and os.path.exists(workspace.path):
            return {
                **state,
                "needs_user_input": False,
            }

        return {
            **state,
            "needs_user_input": True,
            "user_question": "Please provide the path to your ROS workspace (e.g., ~/catkin_ws or ~/ros2_ws):",
            "stage": "workspace_setup",
        }

    def _analyze_environment(self, state: AgentState) -> AgentState:
        """Analyze the ROS workspace environment."""
        workspace = state.get("workspace")
        if not workspace or not workspace.path:
            return state

        workspace_path = workspace.path

        # Detect ROS version
        ros_info = detect_ros_version.invoke({"workspace_path": workspace_path})

        # Analyze workspace
        analysis = analyze_workspace.invoke({"workspace_path": workspace_path})

        # List packages
        packages = list_packages.invoke({"workspace_path": workspace_path})

        # Update workspace info
        updated_workspace = WorkspaceInfo(
            path=workspace_path,
            ros_version=ros_info.get("ros_version", "unknown"),
            ros_distro=ros_info.get("ros_distro", "unknown"),
            gazebo_version=ros_info.get("gazebo_version", "unknown"),
            packages=[p["name"] for p in packages] if packages else [],
            source_files=analysis.get("source_files", {}),
            launch_files=analysis.get("launch_files", []),
            is_analyzed=True
        )

        # Add analysis summary to messages
        analysis_summary = f"""
Workspace Analysis Complete:
- Path: {workspace_path}
- ROS Version: {updated_workspace.ros_version}
- ROS Distribution: {updated_workspace.ros_distro}
- Gazebo Version: {updated_workspace.gazebo_version}
- Packages Found: {len(updated_workspace.packages)}
  {', '.join(updated_workspace.packages[:5])}{'...' if len(updated_workspace.packages) > 5 else ''}
- Launch Files: {len(updated_workspace.launch_files)}
- Source Files: {sum(len(files) for files in updated_workspace.source_files.values())} total
"""

        messages = list(state.get("messages", []))
        messages.append(AIMessage(content=analysis_summary))

        return {
            **state,
            "workspace": updated_workspace,
            "messages": messages,
            "stage": "analysis",
        }

    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate response using LLM with tools."""
        workspace = state.get("workspace")
        workspace_info = "No workspace configured yet."

        if workspace and workspace.is_analyzed:
            workspace_info = f"""
Path: {workspace.path}
ROS Version: {workspace.ros_version}
Distribution: {workspace.ros_distro}
Gazebo: {workspace.gazebo_version}
Packages: {', '.join(workspace.packages[:10])}
"""

        # Build system message
        system_message = SystemMessage(content=SYSTEM_PROMPT.format(
            workspace_info=workspace_info,
            stage=state.get("stage", "init")
        ))

        # Get conversation messages
        messages = [system_message] + list(state.get("messages", []))

        # Generate response
        response = self.llm_with_tools.invoke(messages)

        # Update messages
        updated_messages = list(state.get("messages", []))
        updated_messages.append(response)

        return {
            **state,
            "messages": updated_messages,
            "stage": "coding" if state.get("current_intent") == "generate" else state.get("stage", "init"),
        }

    def _handle_feedback(self, state: AgentState) -> AgentState:
        """Handle user feedback on generated code."""
        messages = state.get("messages", [])
        feedback_history = list(state.get("feedback_history", []))

        # Extract last user message as feedback
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                feedback_history.append(msg.content)
                break

        return {
            **state,
            "feedback_history": feedback_history,
            "stage": "feedback",
        }

    def _ask_user(self, state: AgentState) -> AgentState:
        """Prepare a question for the user."""
        return {
            **state,
            "needs_user_input": True,
        }

    def _route_after_input(self, state: AgentState) -> str:
        """Route after processing input."""
        intent = state.get("current_intent", "general")
        workspace = state.get("workspace")

        if intent == "workspace" or (not workspace or not workspace.is_analyzed):
            return "check_workspace"

        if state.get("needs_user_input"):
            return "ask_user"

        return "generate_response"

    def _route_after_workspace_check(self, state: AgentState) -> str:
        """Route after checking workspace."""
        if state.get("needs_user_input"):
            return "ask_user"
        return "analyze_environment"

    def _route_after_response(self, state: AgentState) -> str:
        """Route after generating response."""
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]

        # Check if the response has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "execute_tools"

        # Check if waiting for feedback
        if state.get("pending_code"):
            return "handle_feedback"

        return "end"

    def _route_after_tools(self, state: AgentState) -> str:
        """Route after tool execution."""
        messages = state.get("messages", [])

        # Check if there are more tool calls needed
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "generate_response"

        return "generate_response"

    def invoke(self, user_input: str, workspace_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke the agent with user input.

        Args:
            user_input: User's message/request
            workspace_path: Optional workspace path to set

        Returns:
            Agent state after processing
        """
        # Initialize state
        workspace = None
        if workspace_path:
            workspace = WorkspaceInfo(path=workspace_path)

        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "workspace": workspace,
            "stage": "init",
            "current_intent": "",
            "pending_code": None,
            "feedback_history": [],
            "error": None,
            "needs_user_input": False,
            "user_question": None,
        }

        # Run the graph
        result = self.graph.invoke(initial_state)
        return result

    def stream(self, user_input: str, workspace_path: Optional[str] = None):
        """
        Stream the agent's response.

        Args:
            user_input: User's message/request
            workspace_path: Optional workspace path

        Yields:
            Intermediate states as the agent processes
        """
        workspace = None
        if workspace_path:
            workspace = WorkspaceInfo(path=workspace_path)

        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "workspace": workspace,
            "stage": "init",
            "current_intent": "",
            "pending_code": None,
            "feedback_history": [],
            "error": None,
            "needs_user_input": False,
            "user_question": None,
        }

        for state in self.graph.stream(initial_state):
            yield state


def create_simulation_graph(
    model_name: str = "gpt-4-turbo-preview",
    api_key: Optional[str] = None
) -> SimulationAgent:
    """
    Factory function to create a simulation agent.

    Args:
        model_name: OpenAI model name
        api_key: OpenAI API key

    Returns:
        Configured SimulationAgent instance
    """
    return SimulationAgent(model_name=model_name, api_key=api_key)
