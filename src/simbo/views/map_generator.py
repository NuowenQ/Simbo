"""
Map Generator page for Simbo.

Converts 2D top-down images to Gazebo world files.
"""

import streamlit as st
import json
from typing import List, Dict, Any
import os

from simbo.views.components import render_back_button

# Import map generator functions with optional dependencies
try:
    from simbo.utils.map_generator import (
        detect_edges,
        detect_shapes,
        generate_gazebo_world
    )
    HAS_MAP_GENERATOR_DEPS = True
except ImportError as e:
    HAS_MAP_GENERATOR_DEPS = False
    # Create placeholder functions
    def detect_edges(image_bytes):
        return {"success": False, "error": "OpenCV not available. Install: pip install opencv-python numpy"}
    
    def detect_shapes(image_bytes):
        return {"success": False, "error": "OpenCV not available. Install: pip install opencv-python numpy"}
    
    def generate_gazebo_world(*args, **kwargs):
        return "<!-- Map generator dependencies not installed -->"


def render_map_generator_page():
    """Render the Map Generator page."""
    # Back button
    render_back_button(key="back_from_map_generator")
    
    # Header
    st.markdown('<p class="main-header">🗺️ Map Generator</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Convert 2D top-down images to Gazebo world files</p>',
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Check if dependencies are available (for cv2/numpy)
    try:
        import cv2
        import numpy as np
        has_opencv = True
    except ImportError:
        has_opencv = False
        st.error("⚠️ **Map Generator dependencies not installed**")
        st.markdown("""
        The Map Generator requires additional dependencies to be installed:
        
        ```bash
        pip install opencv-python numpy pillow
        ```
        
        Or install all requirements:
        ```bash
        pip install -r requirements.txt
        ```
        
        After installing, restart the Streamlit app.
        """)
        st.stop()
    
    # Initialize session state for map data
    if "map_data" not in st.session_state:
        st.session_state.map_data = {
            "walls": [],
            "rooms": [],
            "scale": 20.0,  # pixels per meter
            "image_uploaded": False,
            "detected_edges": [],
            "detected_shapes": []
        }
    
    # File upload section
    st.markdown("### 📤 Upload Map Image")
    uploaded_file = st.file_uploader(
        "Choose a top-down map image (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        key="map_image_upload"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Map Image", use_container_width=True)
        st.session_state.map_data["image_uploaded"] = True
        
        # Detection options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Detect Edges", use_container_width=True):
                with st.spinner("Detecting edges..."):
                    image_bytes = uploaded_file.read()
                    result = detect_edges(image_bytes)
                    
                    if result.get("success"):
                        st.session_state.map_data["detected_edges"] = result.get("edges", [])
                        st.success(f"✅ {result.get('message', 'Edges detected')}")
                        st.info(f"Found {len(result.get('edges', []))} edges. You can use these to create walls.")
                    else:
                        st.error(f"❌ {result.get('error', 'Edge detection failed')}")
        
        with col2:
            if st.button("🔷 Detect Shapes", use_container_width=True):
                with st.spinner("Detecting shapes..."):
                    uploaded_file.seek(0)  # Reset file pointer
                    image_bytes = uploaded_file.read()
                    result = detect_shapes(image_bytes)
                    
                    if result.get("success"):
                        st.session_state.map_data["detected_shapes"] = result.get("shapes", [])
                        st.session_state.map_data["detected_edges"] = result.get("lines", [])
                        st.success(f"✅ {result.get('message', 'Shapes detected')}")
                        
                        shapes = result.get("shapes", [])
                        if shapes:
                            st.info(f"Found {len(shapes)} shapes:")
                            for shape in shapes[:5]:  # Show first 5
                                st.text(f"  - {shape['type']} at ({shape['center']['x']:.1f}, {shape['center']['y']:.1f})")
                    else:
                        st.error(f"❌ {result.get('error', 'Shape detection failed')}")
    
    st.divider()
    
    # Manual wall/room input section
    st.markdown("### ✏️ Manual Map Editing")
    
    # Scale calibration
    scale = st.number_input(
        "Scale (pixels per meter)",
        min_value=1.0,
        max_value=1000.0,
        value=st.session_state.map_data.get("scale", 20.0),
        step=1.0,
        help="How many pixels represent 1 meter in the real world"
    )
    st.session_state.map_data["scale"] = scale
    
    # Walls section
    st.markdown("#### Walls")
    
    # Show detected edges that can be converted to walls
    detected_edges = st.session_state.map_data.get("detected_edges", [])
    if detected_edges:
        st.info(f"💡 {len(detected_edges)} detected edges available. Use 'Add Wall from Edge' to convert them.")
        
        # Allow adding walls from detected edges
        if st.button("➕ Add All Detected Edges as Walls"):
            for edge in detected_edges:
                wall = {
                    "start": edge["start"],
                    "end": edge["end"]
                }
                if wall not in st.session_state.map_data["walls"]:
                    st.session_state.map_data["walls"].append(wall)
            st.success(f"Added {len(detected_edges)} walls")
            st.rerun()
    
    # Manual wall input
    with st.expander("➕ Add Wall Manually"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            wall_start_x = st.number_input("Start X", value=0.0, key="wall_start_x")
        with col2:
            wall_start_y = st.number_input("Start Y", value=0.0, key="wall_start_y")
        with col3:
            wall_end_x = st.number_input("End X", value=100.0, key="wall_end_x")
        with col4:
            wall_end_y = st.number_input("End Y", value=100.0, key="wall_end_y")
        
        if st.button("Add Wall"):
            wall = {
                "start": {"x": wall_start_x, "y": wall_start_y},
                "end": {"x": wall_end_x, "y": wall_end_y}
            }
            st.session_state.map_data["walls"].append(wall)
            st.success("Wall added!")
            st.rerun()
    
    # Display current walls
    walls = st.session_state.map_data.get("walls", [])
    if walls:
        st.markdown(f"**Current Walls: {len(walls)}**")
        for i, wall in enumerate(walls):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.text(f"Wall {i+1}: ({wall['start']['x']:.1f}, {wall['start']['y']:.1f}) → ({wall['end']['x']:.1f}, {wall['end']['y']:.1f})")
            with col2:
                if st.button("Delete", key=f"delete_wall_{i}"):
                    st.session_state.map_data["walls"].pop(i)
                    st.rerun()
    
    st.divider()
    
    # Rooms section
    st.markdown("#### Rooms")
    
    with st.expander("➕ Add Room"):
        room_name = st.text_input("Room Name", value="room", key="room_name")
        st.info("💡 For now, rooms are represented as rectangular areas. Enter the corner points.")
        
        col1, col2 = st.columns(2)
        with col1:
            room_min_x = st.number_input("Min X", value=0.0, key="room_min_x")
            room_min_y = st.number_input("Min Y", value=0.0, key="room_min_y")
        with col2:
            room_max_x = st.number_input("Max X", value=100.0, key="room_max_x")
            room_max_y = st.number_input("Max Y", value=100.0, key="room_max_y")
        
        if st.button("Add Room"):
            room = {
                "name": room_name,
                "points": [
                    {"x": room_min_x, "y": room_min_y},
                    {"x": room_max_x, "y": room_min_y},
                    {"x": room_max_x, "y": room_max_y},
                    {"x": room_min_x, "y": room_max_y}
                ]
            }
            st.session_state.map_data["rooms"].append(room)
            st.success("Room added!")
            st.rerun()
    
    # Display current rooms
    rooms = st.session_state.map_data.get("rooms", [])
    if rooms:
        st.markdown(f"**Current Rooms: {len(rooms)}**")
        for i, room in enumerate(rooms):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"Room {i+1}: {room.get('name', 'room')} ({len(room.get('points', []))} points)")
            with col2:
                if st.button("Delete", key=f"delete_room_{i}"):
                    st.session_state.map_data["rooms"].pop(i)
                    st.rerun()
    
    st.divider()
    
    # World generation section
    st.markdown("### 🎮 Generate Gazebo World")
    
    col1, col2 = st.columns(2)
    with col1:
        wall_height = st.number_input(
            "Wall Height (meters)",
            min_value=0.1,
            max_value=10.0,
            value=3.0,
            step=0.1
        )
    with col2:
        wall_thickness = st.number_input(
            "Wall Thickness (meters)",
            min_value=0.05,
            max_value=2.0,
            value=0.2,
            step=0.05
        )
    
    if st.button("🚀 Generate World File", type="primary", use_container_width=True):
        if not walls and not rooms:
            st.warning("⚠️ Please add at least one wall or room before generating the world file.")
        else:
            with st.spinner("Generating Gazebo world file..."):
                world_content = generate_gazebo_world(
                    walls=walls,
                    rooms=rooms,
                    scale=scale,
                    wall_height=wall_height,
                    wall_thickness=wall_thickness
                )
                
                st.success("✅ World file generated successfully!")
                
                # Display world file content
                with st.expander("📄 View Generated World File"):
                    st.code(world_content, language="xml")
                
                # Download button
                st.download_button(
                    label="📥 Download World File",
                    data=world_content,
                    file_name="generated_world.world",
                    mime="application/xml"
                )
                
                # Option to save to workspace
                if st.session_state.get("workspace_path"):
                    if st.button("💾 Save to Workspace"):
                        workspace_path = st.session_state.workspace_path
                        worlds_package = os.path.join(workspace_path, "src", "simbo_worlds", "worlds")
                        
                        # Create directory if it doesn't exist
                        os.makedirs(worlds_package, exist_ok=True)
                        
                        # Save world file
                        world_file_path = os.path.join(worlds_package, "generated_map.world")
                        with open(world_file_path, 'w') as f:
                            f.write(world_content)
                        
                        st.success(f"✅ World file saved to: {world_file_path}")
                        st.info("💡 You can now launch it with: `ros2 launch simbo_worlds world.launch.py world:=generated_map`")
    
    # Clear data button
    st.divider()
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.map_data = {
            "walls": [],
            "rooms": [],
            "scale": 20.0,
            "image_uploaded": False,
            "detected_edges": [],
            "detected_shapes": []
        }
        st.rerun()
