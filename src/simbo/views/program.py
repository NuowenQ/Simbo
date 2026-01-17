"""
Program page for Simbo.

AI-powered coding assistant interface with chat functionality.
"""

import streamlit as st

from simbo.views.components import (
    render_back_button,
    render_status_indicators,
    render_quick_actions,
    render_chat,
)


def render_program_page():
    """Render the Program page (coding agent interface)."""
    # Back button
    render_back_button(key="back_from_program")

    # Header
    st.markdown('<p class="main-header">💻 Program</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-powered Coding Assistant for ROS/Gazebo Development</p>',
        unsafe_allow_html=True
    )

    # Status indicators
    render_status_indicators()

    st.divider()

    # Quick actions (only if workspace is set)
    if st.session_state.workspace_path:
        render_quick_actions()
        st.divider()

    # Main chat interface
    render_chat()
