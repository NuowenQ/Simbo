"""
World Design Agent - Autonomous agent for ROS/Gazebo world file retrieval and placement.

This agent is specialized for:
1. Taking natural-language descriptions of simulation worlds
2. Finding the best-matching open-source world files
3. Placing world files in the correct ROS package path
4. Ensuring immediate usability with no manual setup

The agent follows a strict pipeline:
    User Prompt
    → Figure out where the world file should be located
    → Intent and constraint extraction
    → Symbolic filtering
    → Vector search (RAG) on metadata
    → LLM re-ranking
    → Final recommendation and links
    → Write world file to correct path
"""

import os
import re
from typing import Dict, List, Optional, Any, Annotated, Sequence, TypedDict
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from ..utils.state import WorkspaceInfo
from ..tools.workspace_tools import (
    detect_ros_version,
    analyze_workspace,
    list_packages,
)
from ..tools.file_tools import (
    read_file,
    list_directory,
)
from ..tools.world_tools import (
    extract_world_constraints,
    search_world_database,
    get_world_details,
    list_available_worlds,
    find_worlds_package,
    create_worlds_package,
    download_world_file,
    write_world_file,
    validate_world_file,
    extract_world_models,
    download_world_models,
    generate_world_launch_snippet,
    find_simulation_launch_files,
    update_simulation_launch_world,
    update_all_simulation_launch_files,
    retrieve_world_file_from_github,
    track_latest_world,
)


# System prompt for the World Design Agent
WORLD_AGENT_SYSTEM_PROMPT = """You are the World Design Agent, a specialized AI assistant for ROS/Gazebo simulation world management.

You are a "ROS-native asset manager that understands simulation semantics" - NOT just a chatbot that suggests files.

## Your Purpose
- Retrieve open-source Gazebo/Ignition world models that match user descriptions
- Place selected world files in the correct ROS package path
- Ensure worlds are immediately usable with no manual setup required

## Current Workspace
Path: {workspace_path}
ROS Version: {ros_version}
ROS Distribution: {ros_distro}
Existing Packages: {packages}
Gazebo Version: {gazebo_version}

## MANDATORY PIPELINE (Execute in Order)

You MUST follow these steps for EVERY world request:

### Step 1: Determine Target Location
- ALWAYS use the fixed package name: simbo_worlds
- If simbo_worlds package doesn't exist, create it automatically
- NEVER ask the user where to place files
- NEVER use /tmp, home directories, or .gazebo
- World files MUST be placed in: src/simbo_worlds/worlds/

### Step 2: Extract Constraints
Use `extract_world_constraints` to parse the user's natural language request into:
- Environment type: indoor/outdoor/mixed
- Scale: small/medium/large
- Terrain: flat/uneven/heightmap/multi_level
- Semantic features: office, warehouse, forest, etc.
- Simulator compatibility: gazebo_classic/ignition

### Step 3: Symbolic Filtering + Search
Use `search_world_database` with the extracted constraints to:
- Apply hard filters first (environment, scale, terrain)
- Then perform semantic search on filtered results
- Get ranked candidate worlds

### Step 4: Re-rank and Select
- Analyze the top candidates
- Select the ONE best match based on:
  - Semantic alignment with user request
  - Complexity appropriateness
  - Simulator compatibility
  - License compatibility

### Step 5: Create Package and Place World File
Use `download_world_file` or `write_world_file` to:
- ALWAYS use the fixed package name: simbo_worlds
- Create simbo_worlds package if it doesn't exist (Python-only, NO C++/CMake):
  ```
  simbo_worlds/
  ├── package.xml
  ├── setup.py
  ├── setup.cfg
  ├── resource/
  │   └── simbo_worlds
  ├── worlds/
  │   └── <name>.world  (your world file)
  └── models/
      └── <model_name>/  (required models)
  ```
- Place the world file in: simbo_worlds/worlds/<name>.world
- The tool handles package creation automatically
- After placing the world file, use `track_latest_world` to mark it as the latest

### Step 5.5: Download Required Models (CRITICAL - DO NOT SKIP)
World files reference models using model:// URIs. These models MUST be downloaded for the world to load correctly in Gazebo.

Use `download_world_models` immediately after placing the world file:
- Pass the world_file_path from the download result
- Pass the workspace_path
- This downloads all referenced models (ground_plane, sun, willowgarage, etc.) to simbo_worlds/models/
- The launch file sets GAZEBO_MODEL_PATH to include this directory

If you skip this step, Gazebo will show an empty world because it cannot find the required models.

### Step 6: Update Simulation Launch File (MANDATORY - NO MANUAL STEPS)
After placing the world file and tracking it as latest, you MUST automatically update ALL existing simulation launch files:
1. Use `update_all_simulation_launch_files` with:
   - workspace_path set to the workspace path
   - use_latest=True
   - This will automatically find and update ALL launch files at once
2. Alternatively, use `find_simulation_launch_files` then `update_simulation_launch_world` for each file
3. The update function will ALWAYS succeed - it will insert the world path code even if no patterns are found
4. NEVER tell the user to manually configure anything - all updates must be automatic
5. NEVER provide manual update instructions in your response
6. The launch file will read from the tracking metadata to get the latest world automatically

### Step 7: Validate and Report
Use `validate_world_file` to verify the world file, then report:
- Selected world and why it was chosen
- File location (ROS package path)
- Source repository link
- Launch command snippet

## File System Rules (MANDATORY)

✅ World files MUST:
- Live inside a ROS package under <package>/worlds/
- Use relative paths via get_package_share_directory()

❌ NEVER place world files in:
- /tmp
- Home directories (~)
- .gazebo directory
- Absolute paths outside ROS workspace

## CRITICAL: Correct Import Statement (MANDATORY)

When generating or updating launch files, you MUST ALWAYS use the CORRECT import:

✅ CORRECT:
```python
from ament_index.python.packages import get_package_share_directory
```

❌ WRONG (DO NOT USE):
```python
from ament_index_python.packages import get_package_share_directory
```

**The correct import uses a DOT (.) between "ament_index" and "python", NOT an underscore (_).**

Always verify that any launch file code you generate or update uses the correct import statement.

## Package Selection Logic (FIXED)

When placing a world file:
1. ALWAYS use the fixed package name: simbo_worlds
2. Check if simbo_worlds package exists in src/
3. If not found → Create simbo_worlds package automatically
4. NEVER use other package names or ask the user

The simbo_worlds package MUST contain:
- package.xml (with ament_python build type - NO CMake!)
- setup.py (Python-only package for installing worlds)
- setup.cfg (required for ROS2 Python packages)
- resource/simbo_worlds (ament marker file)
- worlds/ directory (where world files are placed)
- .simbo_latest_world (metadata file tracking the latest world)

## Output Format

For every request, provide:

1. **Selected World**: Name and ID
2. **Match Explanation**: Why this world fits the request
3. **File Location**: Full ROS package path
4. **Source Link**: Original repository URL
5. **Launch Command**: Ready-to-use ros2 launch command

Example output:
```
## Selected World: AWS Small House World

**Why Selected**: Best match for "indoor home environment for navigation testing"
- Indoor environment ✓
- Small scale (appropriate for home) ✓
- Flat terrain (suitable for wheeled robots) ✓
- Contains furniture and room layouts ✓

**Package Created**:
```
my_worlds/
├── package.xml
├── CMakeLists.txt
└── worlds/
    └── small_house.world
```

**File Location**:
`my_worlds/worlds/small_house.world`

**Source**: https://github.com/aws-robotics/aws-robomaker-small-house-world

**Launch File Updated**:
Modified `robot_bringup/launch/simulation.launch.py` to use the new world.

**Launch Command**:
```bash
ros2 launch my_worlds world.launch.py world:=small_house
# Or use your existing simulation launch:
ros2 launch robot_bringup simulation.launch.py
```
```

## FORBIDDEN Actions

You MUST NOT:
- Modify robot URDFs
- Modify controllers or navigation configs
- Generate worlds from scratch (retrieve existing ones only)
- Embed raw .world file content in prompts
- Ask the user where to place files (decide automatically)
- Ask the user to manually configure launch files (ALWAYS do it automatically)
- Provide manual update instructions (the update function handles everything automatically)

## Error Handling

If no compatible world is found:
1. Explain what was searched for
2. List the closest matches and why they don't fit
3. Suggest alternative search terms
4. Offer to show all available worlds in a category

NOW: Process the user's world request following the mandatory pipeline above.
"""


class WorldAgentState(TypedDict):
    """State for the World Design Agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    workspace_path: str
    workspace_info: Optional[Dict[str, Any]]
    iteration_count: int
    max_iterations: int

    # World-specific state
    extracted_constraints: Optional[Dict[str, Any]]
    search_results: Optional[List[Dict[str, Any]]]
    selected_world: Optional[Dict[str, Any]]
    world_file_path: Optional[str]
    target_package: Optional[str]

    # Completion tracking
    task_complete: bool
    error: Optional[str]


class WorldDesignAgent:
    """
    Autonomous World Design Agent for retrieving and placing Gazebo worlds.

    This agent follows a deterministic pipeline:
    1. Parse user request → extract constraints
    2. Search world database with symbolic + semantic filtering
    3. Re-rank candidates and select best match
    4. Download/place world file in ROS package
    5. Validate and report results
    """

    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_iterations: int = 10
    ):
        """
        Initialize the World Design Agent.

        Args:
            model_name: OpenAI model to use
            temperature: Model temperature (lower = more deterministic)
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

        # World-specific tools
        self.tools = [
            # Constraint extraction
            extract_world_constraints,
            # World search and retrieval
            search_world_database,
            get_world_details,
            list_available_worlds,
            # Package management
            find_worlds_package,
            create_worlds_package,
            # World file operations
            download_world_file,
            write_world_file,
            validate_world_file,
            # Model extraction and download
            extract_world_models,
            download_world_models,
            generate_world_launch_snippet,
            # Launch file management
            find_simulation_launch_files,
            update_simulation_launch_world,
            update_all_simulation_launch_files,
            # Advanced retrieval
            retrieve_world_file_from_github,
            # Latest world tracking
            track_latest_world,
            # Workspace analysis (read-only)
            detect_ros_version,
            analyze_workspace,
            list_packages,
            read_file,
            list_directory,
        ]

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for world retrieval."""

        workflow = StateGraph(WorldAgentState)

        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("validate", self._validate_completion)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "validate": "validate",
                "end": END,
            }
        )

        # After tools, go back to agent
        workflow.add_edge("tools", "agent")

        # After validation, either continue or end
        workflow.add_conditional_edges(
            "validate",
            self._validation_router,
            {
                "continue": "agent",
                "end": END,
            }
        )

        # Use memory saver for checkpointing
        memory = MemorySaver()

        return workflow.compile(checkpointer=memory)

    def _get_system_message(self, state: WorldAgentState) -> SystemMessage:
        """Generate system message with current workspace context."""
        workspace_info = state.get("workspace_info", {})

        return SystemMessage(content=WORLD_AGENT_SYSTEM_PROMPT.format(
            workspace_path=state.get("workspace_path", "Not set"),
            ros_version=workspace_info.get("ros_version", "unknown"),
            ros_distro=workspace_info.get("ros_distro", "unknown"),
            packages=", ".join(workspace_info.get("packages", [])[:10]) or "None detected",
            gazebo_version=workspace_info.get("gazebo_version", "unknown"),
        ))

    def _agent_node(self, state: WorldAgentState) -> Dict[str, Any]:
        """Main agent node that processes requests and decides actions."""
        messages = list(state.get("messages", []))

        # Add system message at the start if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            system_msg = self._get_system_message(state)
            messages = [system_msg] + messages

        # Invoke LLM with tools
        response = self.llm_with_tools.invoke(messages)

        # Track state updates
        iteration_count = state.get("iteration_count", 0) + 1

        # Extract world-related information from tool calls
        extracted_constraints = state.get("extracted_constraints")
        search_results = state.get("search_results")
        selected_world = state.get("selected_world")
        world_file_path = state.get("world_file_path")
        target_package = state.get("target_package")

        # Check tool calls for state updates
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})

                # Track constraint extraction
                if tool_name == "extract_world_constraints":
                    # Will be populated after tool execution
                    pass

                # Track search results
                elif tool_name == "search_world_database":
                    # Will be populated after tool execution
                    pass

                # Track world file operations
                elif tool_name == "download_world_file":
                    selected_world = {"id": tool_args.get("world_id")}

                elif tool_name == "write_world_file":
                    world_file_path = None  # Will be set from result
                
                # Track latest world tracking
                elif tool_name == "track_latest_world":
                    # Latest world is being tracked
                    pass

        return {
            "messages": [response],
            "iteration_count": iteration_count,
            "extracted_constraints": extracted_constraints,
            "search_results": search_results,
            "selected_world": selected_world,
            "world_file_path": world_file_path,
            "target_package": target_package,
        }

    def _should_continue(self, state: WorldAgentState) -> str:
        """Determine if agent should continue, use tools, or validate."""
        messages = state.get("messages", [])
        iteration_count = state.get("iteration_count", 0)

        # Check max iterations
        if iteration_count >= state.get("max_iterations", self.max_iterations):
            return "end"

        if not messages:
            return "end"

        last_message = messages[-1]

        # If there are tool calls, execute them
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # Check if we should validate (world file was placed)
        world_file_path = state.get("world_file_path")
        if world_file_path and os.path.exists(world_file_path):
            return "validate"

        # Check if the agent indicates completion
        if hasattr(last_message, "content") and last_message.content:
            content = last_message.content.lower()
            completion_indicators = [
                "world file",
                "successfully",
                "placed",
                "downloaded",
                "launch command",
                "ros2 launch",
            ]
            if any(ind in content for ind in completion_indicators):
                return "validate"

        return "end"

    def _validate_completion(self, state: WorldAgentState) -> Dict[str, Any]:
        """Validate that the world file was properly placed."""
        world_file_path = state.get("world_file_path")
        workspace_path = state.get("workspace_path")

        validation_passed = False
        error = None

        if world_file_path:
            # Check if file exists
            if os.path.exists(world_file_path):
                # Validate it's in a proper ROS package
                if workspace_path and "src" in world_file_path:
                    if "/worlds/" in world_file_path:
                        validation_passed = True
                    else:
                        error = "World file not in worlds/ directory"
                else:
                    error = "World file not in ROS workspace src directory"
            else:
                error = f"World file not found: {world_file_path}"
        else:
            # Check messages for world file path
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    # Look for file path patterns
                    path_match = re.search(r'(/[^\s]+/worlds/[^\s]+\.world)', msg.content)
                    if path_match:
                        found_path = path_match.group(1)
                        if os.path.exists(found_path):
                            world_file_path = found_path
                            validation_passed = True
                            break

        return {
            "task_complete": validation_passed,
            "world_file_path": world_file_path,
            "error": error,
        }

    def _validation_router(self, state: WorldAgentState) -> str:
        """Route based on validation results."""
        if state.get("task_complete", False):
            return "end"

        # If validation failed but we have iterations left, continue
        if state.get("iteration_count", 0) < state.get("max_iterations", self.max_iterations):
            return "continue"

        return "end"

    def _analyze_workspace(self, workspace_path: str) -> Dict[str, Any]:
        """Pre-analyze the workspace before starting."""
        if not workspace_path or not os.path.exists(workspace_path):
            return {}

        try:
            ros_info = detect_ros_version.invoke({"workspace_path": workspace_path})
            packages_info = list_packages.invoke({"workspace_path": workspace_path})

            # Check for existing worlds package
            worlds_pkg = find_worlds_package.invoke({"workspace_path": workspace_path})

            return {
                "ros_version": ros_info.get("ros_version", "unknown"),
                "ros_distro": ros_info.get("ros_distro", "unknown"),
                "gazebo_version": ros_info.get("gazebo_version", "unknown"),
                "packages": [p["name"] for p in packages_info] if packages_info else [],
                "existing_worlds_package": worlds_pkg.get("package_name") if worlds_pkg.get("found") else None,
                "existing_worlds": worlds_pkg.get("existing_worlds", []),
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
        Invoke the agent with a world request.

        Args:
            user_input: User's description of the desired world
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
        initial_state: WorldAgentState = {
            "messages": [HumanMessage(content=user_input)],
            "workspace_path": workspace_path or "",
            "workspace_info": workspace_info,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "extracted_constraints": None,
            "search_results": None,
            "selected_world": None,
            "world_file_path": None,
            "target_package": workspace_info.get("existing_worlds_package"),
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
            user_input: User's world request
            workspace_path: Path to the ROS workspace
            thread_id: Thread ID for memory

        Yields:
            State updates as the agent works
        """
        workspace_info = {}
        if workspace_path:
            workspace_info = self._analyze_workspace(workspace_path)

        initial_state: WorldAgentState = {
            "messages": [HumanMessage(content=user_input)],
            "workspace_path": workspace_path or "",
            "workspace_info": workspace_info,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "extracted_constraints": None,
            "search_results": None,
            "selected_world": None,
            "world_file_path": None,
            "target_package": workspace_info.get("existing_worlds_package"),
            "task_complete": False,
            "error": None,
        }

        config = {"configurable": {"thread_id": thread_id}}

        for event in self.graph.stream(initial_state, config, stream_mode="values"):
            yield event


def create_world_agent(
    model_name: str = "gpt-4-turbo-preview",
    api_key: Optional[str] = None,
    max_iterations: int = 10
) -> WorldDesignAgent:
    """
    Factory function to create a World Design Agent.

    Args:
        model_name: OpenAI model name
        api_key: OpenAI API key
        max_iterations: Max iterations for autonomous work

    Returns:
        Configured WorldDesignAgent instance
    """
    return WorldDesignAgent(
        model_name=model_name,
        api_key=api_key,
        max_iterations=max_iterations
    )
