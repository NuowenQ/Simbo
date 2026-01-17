"""
Shared UI components for Simbo pages.

Contains reusable sidebar, status indicators, chat interface, and other common UI elements.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any

import streamlit as st

from simbo.utils.state import WorkspaceInfo
from simbo.tools.workspace_tools import detect_ros_version, analyze_workspace, list_packages


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


def render_status_indicators():
    """Render the common status indicators (API key, workspace, model)."""
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


def render_back_button(key: str):
    """Render a back button that returns to home page."""
    if st.button("← Back to Home", key=key):
        st.session_state.current_page = "home"
        st.rerun()


def _render_workspace_config():
    """Render workspace configuration section (shared between sidebars)."""
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

    return workspace_path


def _render_workspace_info(max_packages: int = 8):
    """Render workspace info display."""
    if st.session_state.workspace_info:
        info = st.session_state.workspace_info
        st.success("Workspace analyzed!")

        st.markdown("### Environment")
        st.markdown(f"**ROS:** {info.ros_version} ({info.ros_distro})")
        st.markdown(f"**Gazebo:** {info.gazebo_version}")

        st.markdown("### Packages")
        if info.packages:
            for pkg in info.packages[:max_packages]:
                st.markdown(f"- `{pkg}`")
            if len(info.packages) > max_packages:
                st.markdown(f"*+{len(info.packages) - max_packages} more*")
        else:
            st.markdown("*No packages found*")


def _render_api_config():
    """Render API key and model configuration."""
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


def render_sidebar():
    """Render the full sidebar with configuration options (for Program page)."""
    with st.sidebar:
        _render_api_config()

        st.divider()

        _render_workspace_config()
        _render_workspace_info(max_packages=8)

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
        _render_api_config()

        st.divider()

        _render_workspace_config()
        _render_workspace_info(max_packages=5)


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
    from simbo.agents.simulation_agent import create_simulation_graph

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

    if not st.session_state.agent:
        if not initialize_agent():
            st.error("Please configure your OpenAI API key in the sidebar.")
            return None

    workspace_path = st.session_state.workspace_path if st.session_state.workspace_path else None

    try:
        iteration_count = 0
        final_response = ""
        tools_executed = []
        processed_message_count = 0

        for state in st.session_state.agent.stream(
            user_input=prompt,
            workspace_path=workspace_path,
            thread_id=st.session_state.thread_id
        ):
            messages = state.get("messages", [])
            current_iteration = state.get("iteration_count", 0)

            if current_iteration > iteration_count:
                iteration_count = current_iteration
                status_container.markdown(
                    f'<span class="iteration-badge">Step {iteration_count}</span>',
                    unsafe_allow_html=True
                )

            new_messages = messages[processed_message_count:]
            processed_message_count = len(messages)

            for msg in new_messages:
                tools_info = extract_tool_info(msg)
                for tool in tools_info:
                    tool_display = format_tool_execution(tool["name"], tool["args"])
                    if tool_display not in tools_executed:
                        tools_executed.append(tool_display)
                        status_container.markdown(
                            f'<div class="tool-execution">{tool_display}</div>',
                            unsafe_allow_html=True
                        )

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

                if hasattr(msg, 'content') and msg.content:
                    is_ai_message = (
                        (hasattr(msg, 'type') and msg.type == 'ai') or
                        (hasattr(msg, 'role') and msg.role == 'assistant')
                    )
                    if is_ai_message:
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
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.markdown(content)

    if prompt := st.chat_input("Tell Simbo what you want to build or modify..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_container = st.container()
            st.markdown("---")
            response_container = st.container()

            with status_container:
                st.markdown('<p class="thinking-indicator">🤔 Analyzing request...</p>',
                            unsafe_allow_html=True)

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
