"""
Home page for Simbo.

Main navigation page with cards for different modules.
"""

import os
import streamlit as st

from simbo.views.components import render_status_indicators


def render_main_page():
    """Render the main home page with navigation cards."""
    # Header
    st.markdown('<p class="main-header">Simbo</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Autonomous AI Assistant for ROS/Gazebo Simulation Development</p>',
        unsafe_allow_html=True
    )

    # Status indicators
    render_status_indicators()

    st.divider()

    # Navigation cards
    st.markdown("### Select a Module")
    st.markdown("")

    # Create 2x2 grid for navigation cards using buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "💻\n\n**Program**\n\nAI-powered coding assistant for ROS development",
            key="btn_program",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.current_page = "program"
            st.rerun()

    with col2:
        if st.button(
            "🦾\n\n**Robot**\n\nConfigure and manage robot models",
            key="btn_robot",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.current_page = "robot"
            st.rerun()

    st.markdown("")  # Spacing

    col3, col4 = st.columns(2)

    with col3:
        if st.button(
            "🌍\n\n**World**\n\nDesign and edit Gazebo simulation worlds",
            key="btn_world",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.current_page = "world"
            st.rerun()

    with col4:
        if st.button(
            "📂\n\n**Workspace Manager**\n\nManage ROS packages and workspace structure",
            key="btn_workspace",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.current_page = "workspace_manager"
            st.rerun()

    # Footer tips
    st.markdown("""
    ---
    **Tips:**
    - Set your ROS workspace path in the sidebar first
    - **Program**: Use the AI coding agent to create controllers, launch files, or any ROS code
    - **Robot**: Configure robot models and URDF files
    - **World**: Design Gazebo simulation environments
    - **Workspace Manager**: Manage packages and build your workspace
    """)
