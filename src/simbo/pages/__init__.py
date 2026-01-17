"""
Simbo Pages Module.

Contains individual page components for the Streamlit UI.
"""

from simbo.pages.home import render_main_page
from simbo.pages.program import render_program_page
from simbo.pages.robot import render_robot_page
from simbo.pages.world import render_world_page
from simbo.pages.workspace_manager import render_workspace_manager_page
from simbo.pages.components import (
    render_sidebar,
    render_home_sidebar,
    render_status_indicators,
    render_back_button,
    render_chat,
    render_quick_actions,
)

__all__ = [
    "render_main_page",
    "render_program_page",
    "render_robot_page",
    "render_world_page",
    "render_workspace_manager_page",
    "render_sidebar",
    "render_home_sidebar",
    "render_status_indicators",
    "render_back_button",
    "render_chat",
    "render_quick_actions",
]
