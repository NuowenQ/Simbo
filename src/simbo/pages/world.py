"""
World page for Simbo.

Gazebo world editor interface (placeholder).
"""

import streamlit as st

from simbo.pages.components import render_back_button


def render_world_page():
    """Render the World editor page (placeholder)."""
    # Back button
    render_back_button(key="back_from_world")

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
