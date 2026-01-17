"""
Simulation Agent - Autonomous LangGraph-based agent for Simbo.

This agent behaves like Cursor/Claude Code - it autonomously:
1. Analyzes the user's request
2. Explores the codebase
3. Plans the implementation
4. Writes/edits files directly
5. Runs commands to verify
6. Iterates until the task is complete
"""

import os
from typing import Dict, List, Optional, Any, Annotated, Sequence, TypedDict
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from ..utils.state import WorkspaceInfo
from ..tools.workspace_tools import (
    detect_ros_version,
    analyze_workspace,
    list_packages,
    read_package_xml,
    find_launch_files,
    find_source_files,
)
from ..tools.file_tools import (
    read_file,
    write_file,
    edit_file,
    insert_at_line,
    delete_lines,
    create_directory,
    list_directory,
    search_in_files,
    copy_file,
    delete_file,
)
from ..tools.shell_tools import (
    run_command,
    run_ros_command,
    build_ros_workspace,
    check_ros_topics,
    check_ros_nodes,
    get_topic_info,
    get_message_type,
)


# System prompt that makes the agent behave autonomously like Cursor
SYSTEM_PROMPT = """You are Simbo, an autonomous AI assistant for ROS/Gazebo robotics simulation development.

You work like Cursor or Claude Code - you don't just give advice, you DIRECTLY MODIFY the user's code and files to accomplish their goals.

## Your Capabilities
- Read and analyze existing code in the workspace
- Write new files (controllers, launch files, configs)
- Edit existing files to add features or fix issues
- Run shell commands to build, test, and verify
- Execute ROS commands to check topics, nodes, etc.

## Current Workspace
Path: {workspace_path}
ROS Version: {ros_version}
ROS Distribution: {ros_distro}
Packages: {packages}

## How You Work

1. **ANALYZE**: First, explore the codebase to understand the existing structure
   - Use read_file to examine relevant files
   - Use search_in_files to find related code
   - Use list_directory to understand project structure

2. **PLAN**: Mentally plan what changes need to be made

3. **IMPLEMENT**: Make the actual changes
   - Use write_file to create new files
   - Use edit_file to modify existing files
   - Create proper directory structure with create_directory

4. **VERIFY**: Check that changes are correct
   - Use run_command to build the workspace
   - Use run_ros_command to test ROS functionality
   - Read back files to verify changes

5. **ITERATE**: If something fails, fix it and try again

## Important Rules

- ALWAYS read a file before editing it
- ALWAYS use absolute paths
- ALWAYS verify changes were applied correctly
- If a build fails, analyze the error and fix it
- Keep iterating until the task is COMPLETE
- Report what files you created/modified to the user

## Code Style

For ROS2 Python:
- Use rclpy for node creation
- Follow ROS2 naming conventions
- Include proper type hints
- Add docstrings

For ROS1 Python:
- Use rospy for node creation
- Follow ROS1 conventions

When you complete a task, summarize:
1. What files were created/modified
2. How to use the new code
3. Any additional steps the user needs to take

NOW: Analyze the user's request and start implementing. Don't just describe what to do - DO IT.
"""


class AgentState(TypedDict):
    """State for the autonomous simulation agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    workspace_path: str
    workspace_info: Optional[Dict[str, Any]]
    iteration_count: int
    max_iterations: int
    files_modified: List[str]
    files_created: List[str]
    task_complete: bool
    error: Optional[str]


class SimulationAgent:
    """Autonomous simulation agent that directly modifies user's code."""

    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_iterations: int = 25
    ):
        """
        Initialize the autonomous simulation agent.

        Args:
            model_name: OpenAI model to use
            temperature: Model temperature
            api_key: OpenAI API key
            max_iterations: Maximum iterations before stopping
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.api_key
        )

        self.max_iterations = max_iterations

        # All available tools
        self.tools = [
            # Workspace analysis
            detect_ros_version,
            analyze_workspace,
            list_packages,
            read_package_xml,
            find_launch_files,
            find_source_files,
            # File operations
            read_file,
            write_file,
            edit_file,
            insert_at_line,
            delete_lines,
            create_directory,
            list_directory,
            search_in_files,
            copy_file,
            delete_file,
            # Shell operations
            run_command,
            run_ros_command,
            build_ros_workspace,
            check_ros_topics,
            check_ros_nodes,
            get_topic_info,
            get_message_type,
        ]

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with autonomous loop."""

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("check_completion", self._check_completion)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add edges - the key is the loop back to agent after tools
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "check_completion": "check_completion",
                "end": END,
            }
        )

        # After tools, always go back to agent (this creates the agentic loop)
        workflow.add_edge("tools", "agent")

        # After checking completion, either continue or end
        workflow.add_conditional_edges(
            "check_completion",
            self._completion_router,
            {
                "continue": "agent",
                "end": END,
            }
        )

        # Use memory saver for checkpointing
        memory = MemorySaver()

        return workflow.compile(checkpointer=memory)

    def _get_system_message(self, state: AgentState) -> SystemMessage:
        """Generate system message with current workspace context."""
        workspace_info = state.get("workspace_info", {})

        return SystemMessage(content=SYSTEM_PROMPT.format(
            workspace_path=state.get("workspace_path", "Not set"),
            ros_version=workspace_info.get("ros_version", "unknown"),
            ros_distro=workspace_info.get("ros_distro", "unknown"),
            packages=", ".join(workspace_info.get("packages", [])[:10]) or "None detected"
        ))

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Main agent node that decides what to do next."""
        messages = list(state.get("messages", []))

        # Add system message at the start if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            system_msg = self._get_system_message(state)
            messages = [system_msg] + messages

        # Invoke LLM with tools
        response = self.llm_with_tools.invoke(messages)

        # Track files modified/created from tool calls
        files_modified = list(state.get("files_modified", []))
        files_created = list(state.get("files_created", []))

        # Increment iteration count
        iteration_count = state.get("iteration_count", 0) + 1

        return {
            "messages": [response],
            "iteration_count": iteration_count,
            "files_modified": files_modified,
            "files_created": files_created,
        }

    def _should_continue(self, state: AgentState) -> str:
        """Determine if agent should continue, use tools, or end."""
        messages = state.get("messages", [])
        iteration_count = state.get("iteration_count", 0)

        # Check max iterations
        if iteration_count >= state.get("max_iterations", self.max_iterations):
            return "end"

        # Get last message
        if not messages:
            return "end"

        last_message = messages[-1]

        # If there are tool calls, execute them
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # If no tool calls, check if task is complete
        return "check_completion"

    def _check_completion(self, state: AgentState) -> Dict[str, Any]:
        """Check if the task appears complete."""
        messages = state.get("messages", [])

        # Look at the last AI message
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content") and last_message.content:
                content = last_message.content.lower()

                # Check for completion indicators
                completion_indicators = [
                    "complete",
                    "finished",
                    "done",
                    "created the",
                    "implemented",
                    "here's a summary",
                    "files created",
                    "files modified",
                ]

                if any(ind in content for ind in completion_indicators):
                    return {"task_complete": True}

        return {"task_complete": False}

    def _completion_router(self, state: AgentState) -> str:
        """Route based on task completion status."""
        if state.get("task_complete", False):
            return "end"

        # If we haven't reached max iterations, continue
        if state.get("iteration_count", 0) < state.get("max_iterations", self.max_iterations):
            # But if the last message had no tool calls and wasn't a completion,
            # we should probably end to avoid loops
            messages = state.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "tool_calls") and not last_message.tool_calls:
                    # Agent responded without tools - likely done
                    return "end"

        return "end"

    def _analyze_workspace(self, workspace_path: str) -> Dict[str, Any]:
        """Pre-analyze the workspace before starting."""
        if not workspace_path or not os.path.exists(workspace_path):
            return {}

        try:
            ros_info = detect_ros_version.invoke({"workspace_path": workspace_path})
            packages_info = list_packages.invoke({"workspace_path": workspace_path})

            return {
                "ros_version": ros_info.get("ros_version", "unknown"),
                "ros_distro": ros_info.get("ros_distro", "unknown"),
                "gazebo_version": ros_info.get("gazebo_version", "unknown"),
                "packages": [p["name"] for p in packages_info] if packages_info else [],
            }
        except Exception as e:
            return {"error": str(e)}

    def invoke(
        self,
        user_input: str,
        workspace_path: Optional[str] = None,
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Invoke the agent with a user request.

        Args:
            user_input: User's request/task
            workspace_path: Path to the ROS workspace
            thread_id: Thread ID for conversation memory

        Returns:
            Final state after agent completes
        """
        # Pre-analyze workspace
        workspace_info = {}
        if workspace_path:
            workspace_info = self._analyze_workspace(workspace_path)

        # Initial state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "workspace_path": workspace_path or "",
            "workspace_info": workspace_info,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "files_modified": [],
            "files_created": [],
            "task_complete": False,
            "error": None,
        }

        # Run the graph
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(initial_state, config)

        return result

    def stream(
        self,
        user_input: str,
        workspace_path: Optional[str] = None,
        thread_id: str = "default"
    ):
        """
        Stream the agent's execution for real-time updates.

        Args:
            user_input: User's request
            workspace_path: Path to the ROS workspace
            thread_id: Thread ID for memory

        Yields:
            State updates as the agent works
        """
        workspace_info = {}
        if workspace_path:
            workspace_info = self._analyze_workspace(workspace_path)

        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "workspace_path": workspace_path or "",
            "workspace_info": workspace_info,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "files_modified": [],
            "files_created": [],
            "task_complete": False,
            "error": None,
        }

        config = {"configurable": {"thread_id": thread_id}}

        for event in self.graph.stream(initial_state, config, stream_mode="values"):
            yield event

    def continue_conversation(
        self,
        user_input: str,
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Continue an existing conversation with new input.

        Args:
            user_input: New user input
            thread_id: Thread ID to continue

        Returns:
            Updated state
        """
        config = {"configurable": {"thread_id": thread_id}}

        # Get current state
        current_state = self.graph.get_state(config)

        if current_state and current_state.values:
            # Add new message to existing conversation
            messages = list(current_state.values.get("messages", []))
            messages.append(HumanMessage(content=user_input))

            # Update state
            update = {
                "messages": messages,
                "task_complete": False,
                "iteration_count": 0,  # Reset iteration count for new task
            }

            result = self.graph.invoke(update, config)
            return result
        else:
            # No existing conversation, start fresh
            return self.invoke(user_input, thread_id=thread_id)


def create_simulation_graph(
    model_name: str = "gpt-4-turbo-preview",
    api_key: Optional[str] = None,
    max_iterations: int = 25
) -> SimulationAgent:
    """
    Factory function to create a simulation agent.

    Args:
        model_name: OpenAI model name
        api_key: OpenAI API key
        max_iterations: Max iterations for autonomous work

    Returns:
        Configured SimulationAgent instance
    """
    return SimulationAgent(
        model_name=model_name,
        api_key=api_key,
        max_iterations=max_iterations
    )
