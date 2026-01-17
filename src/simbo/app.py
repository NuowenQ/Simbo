"""
Simbo - Streamlit UI for the ROS/Gazebo Simulation Assistant.

Run with: streamlit run src/simbo/app.py
"""

import os
import streamlit as st
from pathlib import Path
from typing import Optional

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

# Custom CSS
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
    .workspace-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .code-block {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
    }
    .status-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-connected {
        background-color: #4CAF50;
        color: white;
    }
    .status-disconnected {
        background-color: #f44336;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "workspace_path" not in st.session_state:
        st.session_state.workspace_path = ""

    if "workspace_info" not in st.session_state:
        st.session_state.workspace_info = None

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

    if "model" not in st.session_state:
        st.session_state.model = "gpt-4-turbo-preview"

    if "is_analyzing" not in st.session_state:
        st.session_state.is_analyzing = False


def initialize_agent():
    """Initialize or reinitialize the agent."""
    if st.session_state.api_key:
        try:
            st.session_state.agent = create_simulation_graph(
                model_name=st.session_state.model,
                api_key=st.session_state.api_key
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
        # Detect ROS version
        ros_info = detect_ros_version.invoke({"workspace_path": workspace_path})

        # Analyze workspace
        analysis = analyze_workspace.invoke({"workspace_path": workspace_path})

        # List packages
        packages = list_packages.invoke({"workspace_path": workspace_path})

        workspace_info = WorkspaceInfo(
            path=workspace_path,
            ros_version=ros_info.get("ros_version", "unknown"),
            ros_distro=ros_info.get("ros_distro", "unknown"),
            gazebo_version=ros_info.get("gazebo_version", "unknown"),
            packages=[p["name"] for p in packages] if packages else [],
            source_files=analysis.get("source_files", {}),
            launch_files=analysis.get("launch_files", []),
            is_analyzed=True
        )

        return workspace_info

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
            ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"],
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
            if st.button("📂 Browse", use_container_width=True):
                # Note: Streamlit doesn't have a native folder picker
                # Users need to enter the path manually
                st.info("Please enter the path manually")

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
            st.markdown(f"**ROS Version:** {info.ros_version}")
            st.markdown(f"**Distribution:** {info.ros_distro}")
            st.markdown(f"**Gazebo:** {info.gazebo_version}")

            st.markdown("### Packages")
            if info.packages:
                for pkg in info.packages[:10]:
                    st.markdown(f"- `{pkg}`")
                if len(info.packages) > 10:
                    st.markdown(f"*... and {len(info.packages) - 10} more*")
            else:
                st.markdown("*No packages found*")

            st.markdown("### Statistics")
            total_files = sum(len(files) for files in info.source_files.values())
            st.markdown(f"- **Source files:** {total_files}")
            st.markdown(f"- **Launch files:** {len(info.launch_files)}")

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Help section
        with st.expander("❓ Help"):
            st.markdown("""
            **Getting Started:**
            1. Enter your OpenAI API key
            2. Set your ROS workspace path
            3. Click "Analyze" to scan your workspace
            4. Start chatting with Simbo!

            **Example Commands:**
            - "Generate a velocity controller for my robot"
            - "Create a position controller with PID"
            - "Show me the launch files"
            - "Explain this controller code"
            - "Help me debug my simulation"
            """)


def render_chat():
    """Render the chat interface."""
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)

    # Chat input
    if prompt := st.chat_input("Ask Simbo about your simulation..."):
        # Check if agent is initialized
        if not st.session_state.agent:
            if not initialize_agent():
                st.error("Please configure your OpenAI API key in the sidebar.")
                return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke agent
                    result = st.session_state.agent.invoke(
                        user_input=prompt,
                        workspace_path=st.session_state.workspace_path if st.session_state.workspace_path else None
                    )

                    # Extract response
                    messages = result.get("messages", [])
                    response_content = ""

                    for msg in messages:
                        if hasattr(msg, 'content') and msg.content:
                            # Check if it's an AI message (not user)
                            if hasattr(msg, 'type') and msg.type == 'ai':
                                response_content = msg.content
                            elif not hasattr(msg, 'type'):
                                # Handle different message types
                                response_content = msg.content

                    if not response_content:
                        response_content = "I've processed your request. Is there anything specific you'd like me to help you with?"

                    # Check for code in response
                    if "```" in response_content:
                        st.markdown(response_content)
                    else:
                        st.markdown(response_content)

                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_content
                    })

                    # Check if user input is needed
                    if result.get("needs_user_input") and result.get("user_question"):
                        st.info(result["user_question"])

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def render_quick_actions():
    """Render quick action buttons."""
    st.markdown("### Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🎮 Velocity Controller", use_container_width=True):
            prompt = "Generate a velocity controller for my robot that subscribes to cmd_vel and publishes to the appropriate topics."
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

    with col2:
        if st.button("📍 Position Controller", use_container_width=True):
            prompt = "Generate a position controller that moves the robot to a goal position using PID control."
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

    with col3:
        if st.button("🦾 Joint Controller", use_container_width=True):
            prompt = "Generate a joint trajectory controller for controlling robot arm joints."
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

    with col4:
        if st.button("⌨️ Teleop Controller", use_container_width=True):
            prompt = "Generate a keyboard teleop controller for manually controlling the robot."
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Main content
    st.markdown('<p class="main-header">🤖 Simbo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Your AI Assistant for ROS/Gazebo Simulation Development</p>',
        unsafe_allow_html=True
    )

    # Status indicators
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.api_key:
            st.success("🔑 API Key Configured")
        else:
            st.warning("🔑 API Key Required")

    with col2:
        if st.session_state.workspace_info:
            st.success(f"📁 Workspace: {st.session_state.workspace_info.ros_version}")
        else:
            st.info("📁 No Workspace Set")

    with col3:
        if st.session_state.agent:
            st.success(f"🧠 Model: {st.session_state.model}")
        else:
            st.info("🧠 Agent Not Initialized")

    st.divider()

    # Quick actions
    if st.session_state.workspace_info:
        render_quick_actions()
        st.divider()

    # Chat interface
    render_chat()


if __name__ == "__main__":
    main()
