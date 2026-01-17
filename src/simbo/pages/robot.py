"""
Robot page for Simbo.

Robot configuration and management interface (placeholder).
"""

import streamlit as st

from simbo.pages.components import render_back_button


def render_robot_page():
    """Render the Robot configuration page (placeholder)."""
    # Back button
    render_back_button(key="back_from_robot")

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
