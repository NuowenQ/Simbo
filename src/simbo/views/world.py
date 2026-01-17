"""
World page for Simbo.

Gazebo world design agent interface with chat-based world retrieval.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any

import streamlit as st

from simbo.views.components import (
    render_back_button,
    render_status_indicators,
    extract_tool_info,
    format_tool_execution,
)
from simbo.data.world_database import (
    get_world_database,
    EnvironmentType,
    search_worlds_by_text,
)


def _render_world_sidebar():
    """Render the world page sidebar with world-specific additions."""
    with st.sidebar:
        # World database stats
        st.markdown("## 📊 World Database")
        worlds = get_world_database()
        indoor = sum(1 for w in worlds if w.environment_type == EnvironmentType.INDOOR)
        outdoor = sum(1 for w in worlds if w.environment_type == EnvironmentType.OUTDOOR)

        st.metric("Total Worlds", len(worlds))
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indoor", indoor)
        with col2:
            st.metric("Outdoor", outdoor)

        st.divider()

        # Session actions
        if st.button("🏠 Back to Home", use_container_width=True, key="world_back_home"):
            st.session_state.current_page = "home"
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True, key="world_clear_chat"):
                st.session_state.world_messages = []
                st.rerun()
        with col2:
            if st.button("🔄 New Session", use_container_width=True, key="world_new_session"):
                st.session_state.world_thread_id = f"world_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.world_messages = []
                st.rerun()


def _initialize_world_session_state():
    """Initialize session state for world page."""
    if "world_messages" not in st.session_state:
        st.session_state.world_messages = []

    if "world_thread_id" not in st.session_state:
        st.session_state.world_thread_id = f"world_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if "world_agent" not in st.session_state:
        st.session_state.world_agent = None

    if "world_files_placed" not in st.session_state:
        st.session_state.world_files_placed = []


def _format_world_tool_execution(tool_name: str, args: Dict) -> str:
    """Format world-specific tool execution for display."""
    if tool_name == "extract_world_constraints":
        return "🔍 Analyzing request constraints..."
    elif tool_name == "search_world_database":
        query = args.get("query", "")
        return f"🔎 Searching worlds: {query[:40]}..." if len(query) > 40 else f"🔎 Searching: {query}"
    elif tool_name == "get_world_details":
        return f"📋 Getting details: {args.get('world_id', 'unknown')}"
    elif tool_name == "list_available_worlds":
        return "📜 Listing available worlds..."
    elif tool_name == "find_worlds_package":
        return "📦 Finding worlds package..."
    elif tool_name == "create_worlds_package":
        return f"📦 Creating package: {args.get('package_name', 'unknown')}"
    elif tool_name == "download_world_file":
        return f"⬇️ Downloading world: {args.get('world_id', 'unknown')}"
    elif tool_name == "write_world_file":
        return f"📝 Writing world file: {args.get('world_name', 'unknown')}"
    elif tool_name == "validate_world_file":
        return "✅ Validating world file..."
    elif tool_name == "generate_world_launch_snippet":
        return "🚀 Generating launch snippet..."
    else:
        return format_tool_execution(tool_name, args)


def _render_world_agent_streaming(prompt: str, status_container, response_container):
    """Stream the world agent's response with real-time updates."""
    from simbo.agents.world_agent import create_world_agent

    def initialize_world_agent() -> bool:
        """Initialize or reinitialize the world agent."""
        if st.session_state.api_key:
            try:
                st.session_state.world_agent = create_world_agent(
                    model_name=st.session_state.model,
                    api_key=st.session_state.api_key,
                    max_iterations=10
                )
                return True
            except Exception as e:
                st.error(f"Failed to initialize world agent: {e}")
                return False
        return False

    if not st.session_state.world_agent:
        if not initialize_world_agent():
            st.error("Please configure your OpenAI API key in the sidebar.")
            return None

    workspace_path = st.session_state.workspace_path if st.session_state.workspace_path else None

    try:
        iteration_count = 0
        final_response = ""
        tools_executed = []
        processed_message_count = 0

        for state in st.session_state.world_agent.stream(
            user_input=prompt,
            workspace_path=workspace_path,
            thread_id=st.session_state.world_thread_id
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
                    tool_display = _format_world_tool_execution(tool["name"], tool["args"])
                    if tool_display not in tools_executed:
                        tools_executed.append(tool_display)
                        status_container.markdown(
                            f'<div class="tool-execution">{tool_display}</div>',
                            unsafe_allow_html=True
                        )

                        # Track world files placed
                        if tool["name"] == "download_world_file":
                            world_id = tool["args"].get("world_id", "")
                            if world_id and world_id not in st.session_state.world_files_placed:
                                st.session_state.world_files_placed.append(world_id)

                        elif tool["name"] == "write_world_file":
                            world_name = tool["args"].get("world_name", "")
                            if world_name and world_name not in st.session_state.world_files_placed:
                                st.session_state.world_files_placed.append(world_name)

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


def _render_world_chat():
    """Render the world chat interface."""
    for message in st.session_state.world_messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.markdown(content)

    if prompt := st.chat_input("Describe the world you need (e.g., 'indoor warehouse for forklift navigation')..."):
        st.session_state.world_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_container = st.container()
            st.markdown("---")
            response_container = st.container()

            with status_container:
                st.markdown('<p class="thinking-indicator">🌍 Searching world database...</p>',
                            unsafe_allow_html=True)

            response = _render_world_agent_streaming(
                prompt,
                status_container,
                response_container
            )

            if response:
                st.session_state.world_messages.append({
                    "role": "assistant",
                    "content": response
                })
            else:
                fallback = "I encountered an issue processing your request. Please try again."
                st.session_state.world_messages.append({
                    "role": "assistant",
                    "content": fallback
                })


def _render_quick_world_actions():
    """Render quick action buttons for common world requests."""
    st.markdown("### 🚀 Quick World Selection")

    col1, col2, col3, col4 = st.columns(4)

    actions = [
        ("🏠 House", "I need a small indoor house world for home robot navigation testing"),
        ("🏭 Warehouse", "I need a warehouse world with shelves and aisles for logistics robot simulation"),
        ("🏢 Office", "I need an office environment with desks and corridors for indoor navigation"),
        ("🌳 Outdoor", "I need an outdoor world with terrain for outdoor robot testing"),
    ]

    for col, (label, prompt) in zip([col1, col2, col3, col4], actions):
        with col:
            if st.button(label, use_container_width=True, key=f"world_quick_{label}"):
                st.session_state.world_messages.append({"role": "user", "content": prompt})
                st.rerun()


def _render_world_browser():
    """Render a simple world browser for exploring available worlds."""
    with st.expander("📚 Browse Available Worlds", expanded=False):
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            env_filter = st.selectbox(
                "Environment",
                ["All", "Indoor", "Outdoor", "Mixed"],
                key="world_browser_env"
            )
        with col2:
            search_query = st.text_input(
                "Search",
                placeholder="e.g., warehouse, office, forest",
                key="world_browser_search"
            )

        # Get filtered worlds
        worlds = get_world_database()

        if env_filter != "All":
            env_map = {
                "Indoor": EnvironmentType.INDOOR,
                "Outdoor": EnvironmentType.OUTDOOR,
                "Mixed": EnvironmentType.MIXED,
            }
            worlds = [w for w in worlds if w.environment_type == env_map[env_filter]]

        if search_query:
            worlds = search_worlds_by_text(search_query, limit=20)

        # Display worlds in a grid
        for i in range(0, len(worlds), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(worlds):
                    world = worlds[idx]
                    with col:
                        st.markdown(f"**{world.name}**")
                        st.caption(f"{world.environment_type.value} | {world.scale.value} | {world.terrain_type.value}")
                        st.markdown(f"_{world.description[:80]}..._" if len(world.description) > 80 else f"_{world.description}_")

                        # Quick select button
                        if st.button(f"Select", key=f"select_{world.id}"):
                            request = f"I want the {world.name} world ({world.id})"
                            st.session_state.world_messages.append({"role": "user", "content": request})
                            st.rerun()

                        st.markdown("---")


def render_world_page():
    """Render the World Design Agent page."""
    # Initialize session state
    _initialize_world_session_state()

    # Render world-specific sidebar additions (shared sidebar is rendered by app.py)
    _render_world_sidebar()

    # Header
    st.markdown('<p class="main-header">🌍 World Design Agent</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Retrieve and place Gazebo worlds for your ROS simulation</p>',
        unsafe_allow_html=True
    )

    # Status indicators
    render_status_indicators()

    st.divider()

    # Check if workspace is set
    if not st.session_state.workspace_path:
        st.warning("⚠️ Please set your ROS workspace path in the sidebar to place world files.")

    # Quick actions
    _render_quick_world_actions()

    st.divider()

    # World browser
    _render_world_browser()

    st.divider()

    # Main chat interface
    st.markdown("### 💬 World Request")
    st.markdown("_Describe the simulation world you need, and I'll find and place the best match._")

    _render_world_chat()

    # Show placed worlds
    if st.session_state.world_files_placed:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🌍 Worlds Placed")
        for world in st.session_state.world_files_placed[-5:]:
            st.sidebar.markdown(f"- `{world}`")
