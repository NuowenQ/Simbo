"""
Simbo - Streamlit UI for the ROS/Gazebo Simulation Assistant.

This UI shows real-time progress as the autonomous agent works,
including file changes, tool executions, and final results.

Run with: streamlit run src/simbo/app.py
"""

import os
import time
import streamlit as st
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from simbo.agents.simulation_agent import SimulationAgent, create_simulation_graph
from simbo.utils.state import WorkspaceInfo
from simbo.tools.workspace_tools import detect_ros_version, analyze_workspace, list_packages


# Page configuration
st.set_page_config(
    page_title="Simbo - ROS/Gazebo Simulation Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .tool-execution {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.85rem;
        margin: 0.25rem 0;
    }
    .file-created {
        background-color: #1b4332;
        color: #95d5b2;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
    .file-modified {
        background-color: #3d2914;
        color: #f9c74f;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
    .iteration-badge {
        background-color: #4361ee;
        color: white;
        padding: 0.1rem 0.4rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
    }
    .thinking-indicator {
        color: #888;
        font-style: italic;
    }
    /* Navigation card styles */
    .nav-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .nav-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    .nav-card-program {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .nav-card-robot {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .nav-card-world {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .nav-card-workspace {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    .nav-card-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .nav-card-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .nav-card-desc {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
    }
    .main-page-container {
        padding: 2rem 0;
    }
    .back-button {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "messages": [],
        "workspace_path": "",
        "workspace_info": None,
        "agent": None,
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": "gpt-4-turbo-preview",
        "thread_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "files_created": [],
        "files_modified": [],
        "is_processing": False,
        "current_page": "home",  # home, program, robot, world, workspace_manager
        "app_version": "2.0",  # Version to track UI updates
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Force reset to home page if app version changed (new UI)
    if st.session_state.get("app_version") != "2.0":
        st.session_state.app_version = "2.0"
        st.session_state.current_page = "home"


def initialize_agent() -> bool:
    """Initialize or reinitialize the agent."""
    if st.session_state.api_key:
        try:
            st.session_state.agent = create_simulation_graph(
                model_name=st.session_state.model,
                api_key=st.session_state.api_key,
                max_iterations=25
            )
            return True
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            return False
    return False


def analyze_workspace_ui(workspace_path: str) -> Optional[WorkspaceInfo]:
    """Analyze workspace and return info."""
    if not workspace_path or not os.path.exists(workspace_path):
        return None

    try:
        ros_info = detect_ros_version.invoke({"workspace_path": workspace_path})
        analysis = analyze_workspace.invoke({"workspace_path": workspace_path})
        packages = list_packages.invoke({"workspace_path": workspace_path})

        return WorkspaceInfo(
            path=workspace_path,
            ros_version=ros_info.get("ros_version", "unknown"),
            ros_distro=ros_info.get("ros_distro", "unknown"),
            gazebo_version=ros_info.get("gazebo_version", "unknown"),
            packages=[p["name"] for p in packages] if packages else [],
            source_files=analysis.get("source_files", {}),
            launch_files=analysis.get("launch_files", []),
            is_analyzed=True
        )
    except Exception as e:
        st.error(f"Error analyzing workspace: {e}")
        return None


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_key,
            type="password",
            help="Enter your OpenAI API key"
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.session_state.agent = None

        # Model selection
        model = st.selectbox(
            "Model",
            ["gpt-4-turbo-preview", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            index=0,
            help="Select the OpenAI model to use"
        )
        if model != st.session_state.model:
            st.session_state.model = model
            st.session_state.agent = None

        st.divider()

        # Workspace configuration
        st.markdown("## 📁 ROS Workspace")

        workspace_path = st.text_input(
            "Workspace Path",
            value=st.session_state.workspace_path,
            placeholder="/home/user/catkin_ws",
            help="Path to your ROS workspace"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📂 Set Path", use_container_width=True):
                if workspace_path:
                    st.session_state.workspace_path = workspace_path

        with col2:
            analyze_clicked = st.button(
                "🔍 Analyze",
                use_container_width=True,
                disabled=not workspace_path
            )

        if analyze_clicked and workspace_path:
            st.session_state.workspace_path = workspace_path
            with st.spinner("Analyzing workspace..."):
                st.session_state.workspace_info = analyze_workspace_ui(workspace_path)

        # Display workspace info
        if st.session_state.workspace_info:
            info = st.session_state.workspace_info
            st.success("Workspace analyzed!")

            st.markdown("### Environment")
            st.markdown(f"**ROS:** {info.ros_version} ({info.ros_distro})")
            st.markdown(f"**Gazebo:** {info.gazebo_version}")

            st.markdown("### Packages")
            if info.packages:
                for pkg in info.packages[:8]:
                    st.markdown(f"- `{pkg}`")
                if len(info.packages) > 8:
                    st.markdown(f"*+{len(info.packages) - 8} more*")
            else:
                st.markdown("*No packages found*")

        st.divider()

        # Files changed in this session
        if st.session_state.files_created or st.session_state.files_modified:
            st.markdown("## 📝 Files Changed")

            if st.session_state.files_created:
                st.markdown("**Created:**")
                for f in st.session_state.files_created[-5:]:
                    st.markdown(f'<span class="file-created">+ {os.path.basename(f)}</span>',
                                unsafe_allow_html=True)

            if st.session_state.files_modified:
                st.markdown("**Modified:**")
                for f in st.session_state.files_modified[-5:]:
                    st.markdown(f'<span class="file-modified">~ {os.path.basename(f)}</span>',
                                unsafe_allow_html=True)

            st.divider()

        # Actions
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.files_created = []
                st.session_state.files_modified = []
                st.rerun()

        with col2:
            if st.button("🔄 New Session", use_container_width=True):
                st.session_state.thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.messages = []
                st.rerun()


def render_home_sidebar():
    """Render a simplified sidebar for the home page."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_key,
            type="password",
            help="Enter your OpenAI API key"
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.session_state.agent = None

        # Model selection
        model = st.selectbox(
            "Model",
            ["gpt-4-turbo-preview", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            index=0,
            help="Select the OpenAI model to use"
        )
        if model != st.session_state.model:
            st.session_state.model = model
            st.session_state.agent = None

        st.divider()

        # Workspace configuration
        st.markdown("## 📁 ROS Workspace")

        workspace_path = st.text_input(
            "Workspace Path",
            value=st.session_state.workspace_path,
            placeholder="/home/user/catkin_ws",
            help="Path to your ROS workspace"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📂 Set Path", use_container_width=True):
                if workspace_path:
                    st.session_state.workspace_path = workspace_path

        with col2:
            analyze_clicked = st.button(
                "🔍 Analyze",
                use_container_width=True,
                disabled=not workspace_path
            )

        if analyze_clicked and workspace_path:
            st.session_state.workspace_path = workspace_path
            with st.spinner("Analyzing workspace..."):
                st.session_state.workspace_info = analyze_workspace_ui(workspace_path)

        # Display workspace info
        if st.session_state.workspace_info:
            info = st.session_state.workspace_info
            st.success("Workspace analyzed!")

            st.markdown("### Environment")
            st.markdown(f"**ROS:** {info.ros_version} ({info.ros_distro})")
            st.markdown(f"**Gazebo:** {info.gazebo_version}")

            st.markdown("### Packages")
            if info.packages:
                for pkg in info.packages[:5]:
                    st.markdown(f"- `{pkg}`")
                if len(info.packages) > 5:
                    st.markdown(f"*+{len(info.packages) - 5} more*")
            else:
                st.markdown("*No packages found*")


def render_main_page():
    """Render the main home page with navigation cards."""
    # Header
    st.markdown('<p class="main-header">🤖 Simbo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Autonomous AI Assistant for ROS/Gazebo Simulation Development</p>',
        unsafe_allow_html=True
    )

    # Status indicators
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.api_key:
            st.success("🔑 API Key Set")
        else:
            st.error("🔑 API Key Required")

    with col2:
        if st.session_state.workspace_path:
            st.success(f"📁 {os.path.basename(st.session_state.workspace_path)}")
        else:
            st.warning("📁 Set Workspace Path")

    with col3:
        st.info(f"🧠 {st.session_state.model}")

    st.divider()

    # Navigation cards
    st.markdown("### Select a Module")
    st.markdown("")

    # Create 2x2 grid for navigation cards
    col1, col2 = st.columns(2)

    with col1:
        # Program card
        st.markdown("""
        <div class="nav-card nav-card-program">
            <div class="nav-card-icon">💻</div>
            <p class="nav-card-title">Program</p>
            <p class="nav-card-desc">AI-powered coding assistant for ROS development</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Program", key="btn_program", use_container_width=True):
            st.session_state.current_page = "program"
            st.rerun()

    with col2:
        # Robot card
        st.markdown("""
        <div class="nav-card nav-card-robot">
            <div class="nav-card-icon">🦾</div>
            <p class="nav-card-title">Robot</p>
            <p class="nav-card-desc">Configure and manage robot models</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Robot", key="btn_robot", use_container_width=True):
            st.session_state.current_page = "robot"
            st.rerun()

    st.markdown("")  # Spacing

    col3, col4 = st.columns(2)

    with col3:
        # World card
        st.markdown("""
        <div class="nav-card nav-card-world">
            <div class="nav-card-icon">🌍</div>
            <p class="nav-card-title">World</p>
            <p class="nav-card-desc">Design and edit Gazebo simulation worlds</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open World", key="btn_world", use_container_width=True):
            st.session_state.current_page = "world"
            st.rerun()

    with col4:
        # Workspace Manager card
        st.markdown("""
        <div class="nav-card nav-card-workspace">
            <div class="nav-card-icon">📂</div>
            <p class="nav-card-title">Workspace Manager</p>
            <p class="nav-card-desc">Manage ROS packages and workspace structure</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Workspace Manager", key="btn_workspace", use_container_width=True):
            st.session_state.current_page = "workspace_manager"
            st.rerun()

    # Footer tips
    st.markdown("""
    ---
    **💡 Tips:**
    - Set your ROS workspace path in the sidebar first
    - **Program**: Use the AI coding agent to create controllers, launch files, or any ROS code
    - **Robot**: Configure robot models and URDF files
    - **World**: Design Gazebo simulation environments
    - **Workspace Manager**: Manage packages and build your workspace
    """)


def render_program_page():
    """Render the Program page (coding agent interface)."""
    # Back button
    if st.button("← Back to Home", key="back_from_program"):
        st.session_state.current_page = "home"
        st.rerun()

    # Header
    st.markdown('<p class="main-header">💻 Program</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-powered Coding Assistant for ROS/Gazebo Development</p>',
        unsafe_allow_html=True
    )

    # Status indicators
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.api_key:
            st.success("🔑 API Key Set")
        else:
            st.error("🔑 API Key Required")

    with col2:
        if st.session_state.workspace_path:
            st.success(f"📁 {os.path.basename(st.session_state.workspace_path)}")
        else:
            st.warning("📁 Set Workspace Path")

    with col3:
        st.info(f"🧠 {st.session_state.model}")

    st.divider()

    # Quick actions (only if workspace is set)
    if st.session_state.workspace_path:
        render_quick_actions()
        st.divider()

    # Main chat interface
    render_chat()


def render_robot_page():
    """Render the Robot configuration page (placeholder)."""
    # Back button
    if st.button("← Back to Home", key="back_from_robot"):
        st.session_state.current_page = "home"
        st.rerun()

    st.markdown('<p class="main-header">🦾 Robot</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Configure and Manage Robot Models</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.info("🚧 This module is under development. Robot configuration features coming soon!")

    st.markdown("""
    ### Planned Features:
    - URDF/Xacro robot model browser
    - Robot visualization preview
    - Joint and link configuration
    - Sensor configuration
    - Robot model import/export
    """)


def render_world_page():
    """Render the World editor page (placeholder)."""
    # Back button
    if st.button("← Back to Home", key="back_from_world"):
        st.session_state.current_page = "home"
        st.rerun()

    st.markdown('<p class="main-header">🌍 World</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Design and Edit Gazebo Simulation Worlds</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.info("🚧 This module is under development. World editor features coming soon!")

    st.markdown("""
    ### Planned Features:
    - Visual world editor
    - Model library browser
    - Physics configuration
    - Lighting and environment settings
    - World file import/export
    """)


def render_workspace_manager_page():
    """Render the Workspace Manager page (placeholder)."""
    # Back button
    if st.button("← Back to Home", key="back_from_workspace"):
        st.session_state.current_page = "home"
        st.rerun()

    st.markdown('<p class="main-header">📂 Workspace Manager</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Manage ROS Packages and Workspace Structure</p>',
        unsafe_allow_html=True
    )

    st.divider()

    # Show workspace info if available
    if st.session_state.workspace_info:
        info = st.session_state.workspace_info

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Environment Info")
            st.markdown(f"**Path:** `{info.path}`")
            st.markdown(f"**ROS Version:** {info.ros_version}")
            st.markdown(f"**ROS Distro:** {info.ros_distro}")
            st.markdown(f"**Gazebo Version:** {info.gazebo_version}")

        with col2:
            st.markdown("### Packages")
            if info.packages:
                for pkg in info.packages:
                    st.markdown(f"- `{pkg}`")
            else:
                st.markdown("*No packages found*")

        st.divider()

    st.info("🚧 Additional workspace management features coming soon!")

    st.markdown("""
    ### Planned Features:
    - Package creation wizard
    - Dependency management
    - Build and compile tools
    - Package templates
    - Catkin/Colcon workspace tools
    """)


def extract_tool_info(message) -> List[Dict[str, Any]]:
    """Extract tool call information from a message."""
    tools_used = []

    if hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            tool_info = {
                "name": tool_call.get("name", "unknown"),
                "args": tool_call.get("args", {}),
            }
            tools_used.append(tool_info)

    return tools_used


def format_tool_execution(tool_name: str, args: Dict) -> str:
    """Format tool execution for display."""
    # Simplify display based on tool type
    if tool_name == "write_file":
        return f"📝 Writing file: {args.get('file_path', 'unknown')}"
    elif tool_name == "edit_file":
        return f"✏️ Editing file: {args.get('file_path', 'unknown')}"
    elif tool_name == "read_file":
        return f"📖 Reading: {args.get('file_path', 'unknown')}"
    elif tool_name == "run_command":
        cmd = args.get('command', '')
        return f"⚡ Running: {cmd[:60]}..." if len(cmd) > 60 else f"⚡ Running: {cmd}"
    elif tool_name == "search_in_files":
        return f"🔍 Searching for: {args.get('pattern', 'unknown')}"
    elif tool_name == "list_directory":
        return f"📂 Listing: {args.get('dir_path', 'unknown')}"
    elif tool_name == "create_directory":
        return f"📁 Creating directory: {args.get('dir_path', 'unknown')}"
    elif tool_name == "build_ros_workspace":
        return f"🔨 Building workspace..."
    elif tool_name in ["detect_ros_version", "analyze_workspace", "list_packages"]:
        return f"🔍 Analyzing workspace..."
    else:
        return f"🔧 {tool_name}"


def render_agent_response_streaming(prompt: str, status_container, response_container):
    """Stream the agent's response with real-time updates."""

    if not st.session_state.agent:
        if not initialize_agent():
            st.error("Please configure your OpenAI API key in the sidebar.")
            return None

    workspace_path = st.session_state.workspace_path if st.session_state.workspace_path else None

    try:
        iteration_count = 0
        final_response = ""
        tools_executed = []
        processed_message_count = 0  # Track how many messages we've already processed

        # Stream the agent's execution
        for state in st.session_state.agent.stream(
            user_input=prompt,
            workspace_path=workspace_path,
            thread_id=st.session_state.thread_id
        ):
            messages = state.get("messages", [])
            current_iteration = state.get("iteration_count", 0)

            # Update iteration count
            if current_iteration > iteration_count:
                iteration_count = current_iteration
                status_container.markdown(
                    f'<span class="iteration-badge">Step {iteration_count}</span>',
                    unsafe_allow_html=True
                )

            # Only process NEW messages (ones we haven't seen yet)
            new_messages = messages[processed_message_count:]
            processed_message_count = len(messages)

            for msg in new_messages:
                # Check for tool calls
                tools_info = extract_tool_info(msg)
                for tool in tools_info:
                    tool_display = format_tool_execution(tool["name"], tool["args"])
                    if tool_display not in tools_executed:
                        tools_executed.append(tool_display)
                        status_container.markdown(
                            f'<div class="tool-execution">{tool_display}</div>',
                            unsafe_allow_html=True
                        )

                        # Track file changes
                        if tool["name"] == "write_file":
                            file_path = tool["args"].get("file_path", "")
                            if file_path:
                                if not os.path.exists(file_path):
                                    if file_path not in st.session_state.files_created:
                                        st.session_state.files_created.append(file_path)
                                else:
                                    if file_path not in st.session_state.files_modified:
                                        st.session_state.files_modified.append(file_path)

                        elif tool["name"] == "edit_file":
                            file_path = tool["args"].get("file_path", "")
                            if file_path and file_path not in st.session_state.files_modified:
                                st.session_state.files_modified.append(file_path)

                # Get AI response content (only from new messages)
                if hasattr(msg, 'content') and msg.content:
                    is_ai_message = (
                        (hasattr(msg, 'type') and msg.type == 'ai') or
                        (hasattr(msg, 'role') and msg.role == 'assistant')
                    )
                    if is_ai_message:
                        # This is a new AI message, use it as the latest response
                        final_response = msg.content
                        response_container.markdown(final_response)

        return final_response

    except Exception as e:
        st.error(f"Error during execution: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


def render_chat():
    """Render the chat interface with streaming support."""

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.markdown(content)

    # Chat input
    if prompt := st.chat_input("Tell Simbo what you want to build or modify..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response with streaming
        with st.chat_message("assistant"):
            # Create containers for status and response
            status_container = st.container()
            st.markdown("---")
            response_container = st.container()

            with status_container:
                st.markdown('<p class="thinking-indicator">🤔 Analyzing request...</p>',
                            unsafe_allow_html=True)

            # Stream the response
            response = render_agent_response_streaming(
                prompt,
                status_container,
                response_container
            )

            if response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            else:
                fallback = "I encountered an issue processing your request. Please try again."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": fallback
                })


def render_quick_actions():
    """Render quick action buttons."""
    st.markdown("### 🚀 Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    actions = [
        ("🎮 Velocity Controller", "Create a velocity controller node for my robot that subscribes to cmd_vel topic and controls the robot's movement. Write the actual file to my workspace."),
        ("📍 Position Controller", "Create a position controller that uses PID control to move the robot to goal positions. Write the actual code file to my workspace."),
        ("🦾 Joint Controller", "Create a joint trajectory controller for controlling robot manipulator joints. Write the code to my workspace."),
        ("⌨️ Teleop Node", "Create a keyboard teleop node for manual robot control. Write it to my workspace."),
    ]

    for col, (label, prompt) in zip([col1, col2, col3, col4], actions):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()


def main():
    """Main application entry point."""
    init_session_state()

    # Route to appropriate page based on current_page state
    current_page = st.session_state.current_page

    if current_page == "home":
        # Home page uses simplified sidebar
        render_home_sidebar()
        render_main_page()
    elif current_page == "program":
        # Program page uses full sidebar with chat features
        render_sidebar()
        render_program_page()
    elif current_page == "robot":
        render_home_sidebar()
        render_robot_page()
    elif current_page == "world":
        render_home_sidebar()
        render_world_page()
    elif current_page == "workspace_manager":
        render_home_sidebar()
        render_workspace_manager_page()
    else:
        # Default to home page
        render_home_sidebar()
        render_main_page()


if __name__ == "__main__":
    main()
