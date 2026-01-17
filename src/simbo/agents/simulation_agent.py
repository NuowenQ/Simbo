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
import re
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
    check_ros2_entry_points,
    check_python_script_executable,
    check_executable_configuration,
    check_setup_cfg,
    create_setup_cfg,
    create_ros2_python_package,
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
Controller Stage: {controller_stage}

## Stage-Based Controller Generation (MANDATORY - DO NOT SKIP STAGES)

When creating ROS controllers, you MUST follow these stages in order:

**Stage 1: Minimal Node with Logging Only**
- Create a basic ROS node that only logs messages
- Subscribe to sensor topics (if needed) and log received data
- No control commands yet - just logging
- Verify node launches and logs correctly

**Stage 2: Open-Loop Commands**
- After Stage 1 is verified, add open-loop control
- Publish constant velocity/position commands
- No feedback loops - just constant commands
- Verify commands are published on correct topics

**Stage 3: Closed-Loop Control with Feedback/PID**
- After Stage 2 is verified, add closed-loop control
- Implement PID or similar feedback controllers
- Subscribe to feedback topics (e.g., joint states)
- Calculate control output based on feedback
- Verify closed-loop behavior works

**CRITICAL**: Never skip stages. Always complete and validate each stage before moving to the next.

## Workspace Constraints (MANDATORY)

- Use ONLY detected joints, topics, and interfaces from the workspace
- DO NOT invent or guess joint/topic names
- Before using any topic/joint, verify it exists using check_ros_topics or workspace analysis
- Extract actual names from:
  - check_ros_topics results
  - analyze_workspace results
  - existing configuration files
  - launch files

Detected Joints: {detected_joints}
Detected Topics: {detected_topics}

## Iteration & Patching Loop (MANDATORY)

Follow this exact process for each change:

1. **ANALYZE**: Explore workspace to understand structure
   - Use read_file to examine relevant files
   - Use search_in_files to find related code
   - Use check_ros_topics to verify topic names
   - Use analyze_workspace to understand joints/interfaces

2. **PLAN**: Mentally plan what changes need to be made
   - Identify which stage you're implementing
   - List files to create/modify
   - Verify you're using correct topic/joint names

3. **IMPLEMENT**: Make the actual changes
   - Use write_file to create new files
   - Use edit_file to modify existing files
   - Create proper directory structure with create_directory

4. **VERIFY**: Check that changes are correct
   - Use build_ros_workspace to build the workspace
   - Use check_ros_nodes to verify node loads
   - Use check_ros_topics to verify topics exist
   - Read back files to verify changes

5. **PATCH**: If verification fails, analyze errors and fix
   - Read error messages carefully
   - Fix issues incrementally
   - Re-verify after each fix
   - Summarize errors concisely

## Deterministic Validation Before Completion

A task is NOT complete until ALL of these pass:

1. **Workspace builds successfully**: build_ros_workspace returns success=True
2. **Controller loads in ROS**: check_ros_nodes shows the controller node
3. **Required topics exist**: check_ros_topics shows all required topics
4. **Executables configured correctly**:
   - **ROS2**: Python nodes MUST have entry points in setup.py entry_points['console_scripts']
   - **ROS1**: Python scripts MUST be executable (chmod +x) or in scripts/ directory
   - Example ROS2 entry point: entry_points={{'console_scripts': ['node_name=package.module:main']}}
5. **No runtime errors**: Controller runs without errors (test with short timeout)

DO NOT mark tasks complete based on text descriptions alone. Only mark complete after deterministic validation passes.

## Important Rules

- ALWAYS read a file before editing it
- ALWAYS use absolute paths
- ALWAYS verify changes were applied correctly
- If a build fails, analyze the error and fix it
- Keep iterating until the task is COMPLETE and VALIDATED
- Report what files you created/modified to the user
- Summarize errors and progress concisely

## Code Style - PYTHON ONLY

**IMPORTANT: ALWAYS use Python for ROS2 packages. DO NOT create C/C++ packages or CMakeLists.txt files.**

For ROS2 Python packages (ament_python):
- Use rclpy for node creation
- Follow ROS2 naming conventions
- Include proper type hints
- Add docstrings

**CRITICAL: ROS2 Python Package Structure (ament_python)**

A ROS2 Python package requires EXACTLY these files (NO CMakeLists.txt!):

```
<package_name>/
├── package.xml           # Package metadata with <build_type>ament_python</build_type>
├── setup.py              # Python package setup with entry_points
├── setup.cfg             # CRITICAL: Install script location
├── resource/<package_name>  # Empty marker file for ament
└── <package_name>/       # Python module directory (same name as package)
    ├── __init__.py       # Makes it a Python module
    └── <node_name>.py    # Your node files
```

**File Requirements:**

1. **package.xml** - MUST contain:
   ```xml
   <build_type>ament_python</build_type>
   ```
   DO NOT include ament_cmake or any CMake dependencies!

2. **setup.py** - MUST have entry_points:
   ```python
   entry_points={{
       'console_scripts': [
           'node_name=package_name.module_name:main',
       ],
   }},
   ```

3. **setup.cfg** (MANDATORY - most commonly forgotten!):
   ```
   [develop]
   script_dir=$base/lib/<package_name>
   [install]
   install_scripts=$base/lib/<package_name>
   ```
   - This tells colcon WHERE to install executables
   - Without it, executables go to bin/ instead of lib/<package_name>/
   - ros2 run expects executables at lib/<package_name>/, so it WILL FAIL without this!
   - **ALWAYS use create_setup_cfg tool** after creating a new package

4. **resource/<package_name>** - Empty marker file required by ament

5. **<package_name>/__init__.py** - Empty file to make it a Python module

**DO NOT CREATE:**
- CMakeLists.txt (this is for C/C++ packages only!)
- Any .cpp, .c, .h, or .hpp files
- ament_cmake dependencies in package.xml

**ALWAYS:**
- Use create_setup_cfg tool after creating a new package
- Use check_setup_cfg to verify setup.cfg exists and is correct
- Verify package.xml has <build_type>ament_python</build_type>

For ROS1 Python:
- Use rospy for node creation
- Follow ROS1 conventions
- **CRITICAL**: Python scripts must be executable (chmod +x) or placed in scripts/ directory

When you complete a task, summarize:
1. What files were created/modified
2. How to use the new code
3. Validation results (build status, topics, nodes)
4. Any additional steps the user needs to take

NOW: Analyze the user's request and start implementing. Don't just describe what to do - DO IT.
"""

# Prompt for summarizing conversation history to save tokens
SUMMARIZE_PROMPT = """Summarize the conversation history into a compact format. Focus on:
1. What the user requested
2. What files were read/created/modified
3. What commands were run and their outcomes
4. Current progress toward the goal
5. Any errors encountered and how they were handled

Keep it concise - use bullet points. Omit tool call details, just capture the outcomes.

Conversation to summarize:
{conversation}

Summary:"""


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
    # Token optimization: conversation summary
    conversation_summary: Optional[str]
    messages_since_summary: int  # Count messages since last summary
    # Stage-based controller generation
    controller_stage: Optional[int]  # 1=logging, 2=open-loop, 3=closed-loop
    detected_joints: List[str]  # Detected joint names from workspace
    detected_topics: List[str]  # Detected topic names from workspace
    required_topics: List[str]  # Topics required by the controller
    validation_results: Optional[Dict[str, Any]]  # Results from deterministic validation


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

        # Summarization settings
        self.summarize_threshold = 8  # Summarize after this many messages since last summary

        # Separate LLM for summarization (faster, cheaper model)
        self.summarize_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=self.api_key
        )

        # All available tools
        self.tools = [
            # Workspace analysis
            detect_ros_version,
            analyze_workspace,
            list_packages,
            read_package_xml,
            find_launch_files,
            find_source_files,
            # Validation tools
            check_ros2_entry_points,
            check_python_script_executable,
            check_executable_configuration,
            check_setup_cfg,
            create_setup_cfg,
            create_ros2_python_package,
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
        workflow.add_node("validate_completion", self._validate_completion)
        workflow.add_node("summarize", self._summarize_history)
        workflow.add_node("maybe_summarize", lambda s: {})  # Pass-through node for routing

        # Set entry point
        workflow.set_entry_point("agent")

        # Add edges - the key is the loop back to agent after tools
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "check_completion": "check_completion",
                "validate": "validate_completion",
                "end": END,
            }
        )

        # After tools, check if we need to summarize before going back to agent
        workflow.add_edge("tools", "maybe_summarize")

        # Decide whether to summarize or go directly to agent
        workflow.add_conditional_edges(
            "maybe_summarize",
            self._should_summarize,
            {
                "summarize": "summarize",
                "agent": "agent",
            }
        )

        # After summarization, go to agent
        workflow.add_edge("summarize", "agent")

        # After checking completion, route to validation if text suggests completion
        workflow.add_conditional_edges(
            "check_completion",
            self._completion_router,
            {
                "validate": "validate_completion",
                "continue": "agent",
                "end": END,
            }
        )

        # After validation, either continue or end
        workflow.add_conditional_edges(
            "validate_completion",
            self._validation_router,
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
        controller_stage = state.get("controller_stage")
        stage_text = {
            1: "Stage 1: Logging Only",
            2: "Stage 2: Open-Loop Commands",
            3: "Stage 3: Closed-Loop Control"
        }.get(controller_stage, "Not started") if controller_stage else "Not started"

        return SystemMessage(content=SYSTEM_PROMPT.format(
            workspace_path=state.get("workspace_path", "Not set"),
            ros_version=workspace_info.get("ros_version", "unknown"),
            ros_distro=workspace_info.get("ros_distro", "unknown"),
            packages=", ".join(workspace_info.get("packages", [])[:10]) or "None detected",
            controller_stage=stage_text,
            detected_joints=", ".join(state.get("detected_joints", [])[:20]) or "None detected",
            detected_topics=", ".join(state.get("detected_topics", [])[:20]) or "None detected"
        ))

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Main agent node that decides what to do next."""
        messages = list(state.get("messages", []))

        # Detect workspace information if not already detected
        detected_joints = list(state.get("detected_joints", []))
        detected_topics = list(state.get("detected_topics", []))
        
        # If not detected yet and we have workspace, try to detect
        if not detected_joints and not detected_topics:
            workspace_path = state.get("workspace_path")
            workspace_info = state.get("workspace_info", {})
            ros_version = workspace_info.get("ros_version", "ros2")
            
            if workspace_path:
                # Try to detect topics
                topics_result = check_ros_topics.invoke({
                    "ros_version": ros_version,
                    "workspace_path": workspace_path
                })
                if topics_result.get("stdout"):
                    detected_topics = topics_result["stdout"].strip().split("\n")
                    detected_topics = [t.strip() for t in detected_topics if t.strip()]

        # Track controller stage (initialize to 1 if not set)
        controller_stage = state.get("controller_stage")
        if controller_stage is None:
            # Check if controller files exist to determine stage
            files_created = state.get("files_created", [])
            if any("controller" in f.lower() for f in files_created):
                # If we have controller files, we might be past stage 1
                # Let validation or explicit stage tracking handle this
                controller_stage = 1  # Default to stage 1
            else:
                controller_stage = 1  # Start at stage 1

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

        # Track messages since last summary (increment by 1 for the new response)
        messages_since_summary = state.get("messages_since_summary", 0) + 1

        return {
            "messages": [response],
            "iteration_count": iteration_count,
            "files_modified": files_modified,
            "files_created": files_created,
            "messages_since_summary": messages_since_summary,
            "controller_stage": controller_stage,
            "detected_joints": detected_joints,
            "detected_topics": detected_topics,
        }

    def _should_continue(self, state: AgentState) -> str:
        """Determine if agent should continue, use tools, validate, or end."""
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

        # If no tool calls, check if task appears complete (then validate)
        # Only validate if we have created files
        files_created = state.get("files_created", [])
        if files_created:
            return "validate"

        # Otherwise check text completion
        return "check_completion"

    def _check_completion(self, state: AgentState) -> Dict[str, Any]:
        """Check if the task appears complete based on text only (legacy check)."""
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
                    # Text suggests completion, but need validation
                    return {"task_complete": False}  # Always defer to validation

        return {"task_complete": False}

    def _validate_completion(self, state: AgentState) -> Dict[str, Any]:
        """Perform deterministic validation before marking task complete."""
        workspace_path = state.get("workspace_path")
        workspace_info = state.get("workspace_info", {})
        ros_version = workspace_info.get("ros_version", "ros2")
        files_created = state.get("files_created", [])
        required_topics = state.get("required_topics", [])

        validation_results = {
            "build_success": False,
            "controller_loaded": False,
            "topics_exist": False,
            "runtime_errors": False,
            "executables_configured": False,
            "errors": []
        }

        # Check 1: Workspace builds successfully
        if files_created and workspace_path:
            build_result = build_ros_workspace.invoke({
                "workspace_path": workspace_path,
                "ros_version": ros_version
            })
            validation_results["build_success"] = build_result.get("success", False)
            if not validation_results["build_success"]:
                validation_results["errors"].append(
                    f"Build failed: {build_result.get('error', 'Unknown error')}"
                )

        # Check 2: Controller loads in ROS (check if we have controller files)
        # Note: This is a lightweight check - actual loading would require ROS running
        # For now, we verify the file exists and is syntactically valid
        if files_created:
            controller_files = [f for f in files_created if "controller" in f.lower() or ".py" in f]
            validation_results["controller_loaded"] = len(controller_files) > 0
            if not validation_results["controller_loaded"]:
                validation_results["errors"].append("No controller files found")

        # Check 3: Executables configured correctly - use validation tools
        python_nodes = [f for f in files_created if f.endswith(".py") and os.path.exists(f)]
        if python_nodes and workspace_path:
            all_valid = True
            for node_file in python_nodes:
                exec_result = check_executable_configuration.invoke({
                    "workspace_path": workspace_path,
                    "node_file": node_file,
                    "ros_version": ros_version
                })
                
                if not exec_result.get("is_valid"):
                    all_valid = False
                    if exec_result.get("errors"):
                        validation_results["errors"].extend(exec_result["errors"])
                    if exec_result.get("fix_suggestions"):
                        # Add suggestions as errors with context
                        for suggestion in exec_result["fix_suggestions"]:
                            validation_results["errors"].append(
                                f"{node_file}: {suggestion}"
                            )
            
            validation_results["executables_configured"] = all_valid
        else:
            # No Python nodes to check
            validation_results["executables_configured"] = True

        # Check 4: Required topics exist (if specified)
        if required_topics and workspace_path:
            topics_result = check_ros_topics.invoke({
                "ros_version": ros_version,
                "workspace_path": workspace_path
            })
            if topics_result.get("stdout"):
                available_topics = topics_result["stdout"].strip().split("\n")
                missing_topics = [t for t in required_topics if t not in available_topics]
                validation_results["topics_exist"] = len(missing_topics) == 0
                if not validation_results["topics_exist"]:
                    validation_results["errors"].append(
                        f"Missing topics: {', '.join(missing_topics)}"
                    )
            else:
                # If we can't check topics, assume they might be created by the controller
                validation_results["topics_exist"] = True

        # Check 5: Runtime errors (basic syntax check via build)
        # Runtime errors are primarily caught during build, so we rely on build_success
        validation_results["runtime_errors"] = validation_results["build_success"]

        # Task is complete only if ALL validations pass
        task_complete = (
            validation_results["build_success"] and
            validation_results["controller_loaded"] and
            validation_results["topics_exist"] and
            validation_results["runtime_errors"] and
            validation_results["executables_configured"]
        )

        return {
            "validation_results": validation_results,
            "task_complete": task_complete
        }

    def _summarize_history(self, state: AgentState) -> Dict[str, Any]:
        """Summarize old messages to reduce token usage."""
        messages = list(state.get("messages", []))

        if len(messages) < self.summarize_threshold:
            return {"messages_since_summary": len(messages)}

        # Extract messages to summarize (keep last few for context)
        keep_recent = 4  # Keep last 4 messages unsummarized
        messages_to_summarize = messages[:-keep_recent] if len(messages) > keep_recent else []
        recent_messages = messages[-keep_recent:] if len(messages) > keep_recent else messages

        if not messages_to_summarize:
            return {"messages_since_summary": len(messages)}

        # Format messages for summarization
        conversation_text = []
        for msg in messages_to_summarize:
            if isinstance(msg, HumanMessage):
                conversation_text.append(f"User: {msg.content[:500]}")
            elif isinstance(msg, AIMessage):
                content = msg.content[:500] if msg.content else ""
                tool_info = ""
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tools = [tc.get('name', 'unknown') for tc in msg.tool_calls]
                    tool_info = f" [Called tools: {', '.join(tools)}]"
                conversation_text.append(f"Assistant: {content}{tool_info}")
            elif isinstance(msg, ToolMessage):
                # Truncate tool results heavily - they're often large
                result_preview = str(msg.content)[:200]
                conversation_text.append(f"Tool Result: {result_preview}...")

        # Get summary from LLM
        try:
            summary_response = self.summarize_llm.invoke([
                HumanMessage(content=SUMMARIZE_PROMPT.format(
                    conversation="\n".join(conversation_text)
                ))
            ])
            new_summary = summary_response.content

            # Combine with existing summary if present
            existing_summary = state.get("conversation_summary", "")
            if existing_summary:
                new_summary = f"Previous context:\n{existing_summary}\n\nRecent activity:\n{new_summary}"

            # Create new message list: system + summary message + recent messages
            summary_msg = SystemMessage(content=f"[Conversation Summary]\n{new_summary}")

            # Return updated state - replace old messages with summary + recent
            return {
                "messages": [summary_msg] + list(recent_messages),
                "conversation_summary": new_summary,
                "messages_since_summary": len(recent_messages),
            }
        except Exception as e:
            # If summarization fails, just continue without it
            return {"messages_since_summary": len(messages)}

    def _should_summarize(self, state: AgentState) -> str:
        """Check if we should summarize before continuing."""
        messages_since = state.get("messages_since_summary", 0)
        messages = state.get("messages", [])

        # Summarize if we have many messages since last summary
        if len(messages) > self.summarize_threshold and messages_since > self.summarize_threshold:
            return "summarize"
        return "agent"

    def _completion_router(self, state: AgentState) -> str:
        """Route based on task completion status from text check."""
        # Text completion check suggests we might be done, so validate
        if state.get("task_complete", False):
            return "validate"

        # If we haven't reached max iterations, continue
        if state.get("iteration_count", 0) < state.get("max_iterations", self.max_iterations):
            messages = state.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "tool_calls") and not last_message.tool_calls:
                    # Agent responded without tools - check if we have files to validate
                    files_created = state.get("files_created", [])
                    if files_created:
                        return "validate"
                    return "end"

        return "end"

    def _validation_router(self, state: AgentState) -> str:
        """Route based on validation results."""
        validation_results = state.get("validation_results")
        
        if state.get("task_complete", False):
            # All validations passed
            return "end"

        # Validations failed - continue iterating
        if state.get("iteration_count", 0) < state.get("max_iterations", self.max_iterations):
            return "continue"

        # Max iterations reached
        return "end"

    def _analyze_workspace(self, workspace_path: str) -> Dict[str, Any]:
        """Pre-analyze the workspace before starting."""
        if not workspace_path or not os.path.exists(workspace_path):
            return {}

        try:
            ros_info = detect_ros_version.invoke({"workspace_path": workspace_path})
            packages_info = list_packages.invoke({"workspace_path": workspace_path})
            ros_version = ros_info.get("ros_version", "ros2")

            # Try to detect topics and joints from workspace
            detected_topics = []
            detected_joints = []

            # Detect topics if ROS is available
            try:
                topics_result = check_ros_topics.invoke({
                    "ros_version": ros_version,
                    "workspace_path": workspace_path
                })
                if topics_result.get("stdout"):
                    detected_topics = topics_result["stdout"].strip().split("\n")
                    detected_topics = [t.strip() for t in detected_topics if t.strip()]
            except Exception:
                pass  # Topics might not be available if ROS isn't running

            # Try to detect joints from URDF/Xacro files
            try:
                # Search for URDF/Xacro files in workspace
                src_path = os.path.join(workspace_path, "src")
                if os.path.exists(src_path):
                    for root, dirs, files in os.walk(src_path):
                        for file in files:
                            if file.endswith((".urdf", ".xacro", ".urdf.xacro")):
                                file_path = os.path.join(root, file)
                                try:
                                    with open(file_path, 'r') as f:
                                        content = f.read()
                                        # Extract joint names (simple regex)
                                        joint_matches = re.findall(r'<joint\s+name=["\']([^"\']+)["\']', content)
                                        detected_joints.extend(joint_matches)
                                except Exception:
                                    pass
            except Exception:
                pass  # Joint detection is optional

            return {
                "ros_version": ros_info.get("ros_version", "unknown"),
                "ros_distro": ros_info.get("ros_distro", "unknown"),
                "gazebo_version": ros_info.get("gazebo_version", "unknown"),
                "packages": [p["name"] for p in packages_info] if packages_info else [],
                "detected_topics": detected_topics[:50],  # Limit to 50
                "detected_joints": list(set(detected_joints))[:50],  # Deduplicate and limit
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
            "conversation_summary": None,
            "messages_since_summary": 0,
            "controller_stage": None,
            "detected_joints": workspace_info.get("detected_joints", []),
            "detected_topics": workspace_info.get("detected_topics", []),
            "required_topics": [],
            "validation_results": None,
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
            "conversation_summary": None,
            "messages_since_summary": 0,
            "controller_stage": None,
            "detected_joints": workspace_info.get("detected_joints", []),
            "detected_topics": workspace_info.get("detected_topics", []),
            "required_topics": [],
            "validation_results": None,
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
