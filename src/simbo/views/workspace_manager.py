"""
Workspace Manager page for Simbo.

ROS workspace management interface (placeholder).
"""

import streamlit as st

from simbo.views.components import render_back_button


def render_workspace_manager_page():
    """Render the Workspace Manager page (placeholder)."""
    # Back button
    render_back_button(key="back_from_workspace")

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
