"""
Home page for Simbo.

Main navigation page with cards for different modules.
"""

import os
import streamlit as st

from simbo.pages.components import render_status_indicators


def render_main_page():
    """Render the main home page with navigation cards."""
    # Header
    st.markdown('<p class="main-header">🤖 Simbo</p>', unsafe_allow_html=True)
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
