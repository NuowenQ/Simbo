"""
Simbo - Streamlit UI for the ROS/Gazebo Simulation Assistant.

This UI shows real-time progress as the autonomous agent works,
including file changes, tool executions, and final results.

Run with: streamlit run src/simbo/app.py
"""

import os
import streamlit as st
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from simbo.pages import (
    render_main_page,
    render_program_page,
    render_robot_page,
    render_world_page,
    render_workspace_manager_page,
    render_sidebar,
    render_home_sidebar,
)


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
