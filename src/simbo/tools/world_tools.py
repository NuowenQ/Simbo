"""
Tools for World Design Agent.

This module provides tools for:
- Finding and selecting world files from open-source repositories
- Managing ROS packages for world files
- Downloading and placing world files in correct ROS package paths
"""

import os
import re
import subprocess
import urllib.request
import urllib.error
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from langchain_core.tools import tool

from simbo.data.world_database import (
    WorldMetadata,
    EnvironmentType,
    Scale,
    TerrainType,
    SimulatorCompatibility,
    get_world_database,
    get_world_by_id,
    filter_worlds,
    search_worlds_by_text,
)


# =============================================================================
# Intent Extraction Tools
# =============================================================================

@tool
def extract_world_constraints(user_prompt: str) -> Dict[str, Any]:
    """
    Extract structured constraints from a natural language world description.

    This tool analyzes the user's prompt and extracts:
    - Environment type (indoor/outdoor/mixed)
    - Scale (small/medium/large)
    - Terrain type (flat/uneven/heightmap)
    - Semantic features (office, warehouse, forest, etc.)
    - Simulator compatibility preference

    Args:
        user_prompt: The user's natural language description of the desired world

    Returns:
        Dictionary with extracted constraints
    """
    prompt_lower = user_prompt.lower()

    constraints = {
        "environment_type": None,
        "scale": None,
        "terrain_type": None,
        "semantic_features": [],
        "simulator_compatibility": None,
        "original_prompt": user_prompt,
    }

    # Environment type detection
    indoor_keywords = ["indoor", "inside", "room", "building", "house", "office",
                       "warehouse", "hospital", "store", "cafe", "factory"]
    outdoor_keywords = ["outdoor", "outside", "field", "street", "road", "forest",
                        "city", "urban", "farm", "terrain", "landscape"]

    indoor_count = sum(1 for kw in indoor_keywords if kw in prompt_lower)
    outdoor_count = sum(1 for kw in outdoor_keywords if kw in prompt_lower)

    if indoor_count > outdoor_count:
        constraints["environment_type"] = "indoor"
    elif outdoor_count > indoor_count:
        constraints["environment_type"] = "outdoor"
    elif indoor_count > 0 and outdoor_count > 0:
        constraints["environment_type"] = "mixed"

    # Scale detection
    small_keywords = ["small", "tiny", "compact", "simple", "basic", "minimal"]
    medium_keywords = ["medium", "moderate", "standard", "typical"]
    large_keywords = ["large", "big", "extensive", "complex", "detailed", "city", "warehouse"]

    if any(kw in prompt_lower for kw in small_keywords):
        constraints["scale"] = "small"
    elif any(kw in prompt_lower for kw in large_keywords):
        constraints["scale"] = "large"
    elif any(kw in prompt_lower for kw in medium_keywords):
        constraints["scale"] = "medium"

    # Terrain type detection
    if any(kw in prompt_lower for kw in ["heightmap", "terrain", "hills", "elevation", "slopes"]):
        constraints["terrain_type"] = "heightmap"
    elif any(kw in prompt_lower for kw in ["uneven", "rough", "rocky", "bumpy", "irregular"]):
        constraints["terrain_type"] = "uneven"
    elif any(kw in prompt_lower for kw in ["multi-level", "multi level", "floors", "stories", "levels"]):
        constraints["terrain_type"] = "multi_level"
    elif any(kw in prompt_lower for kw in ["flat", "smooth", "level", "ground"]):
        constraints["terrain_type"] = "flat"

    # Simulator compatibility detection
    if any(kw in prompt_lower for kw in ["ignition", "ign", "gz-sim", "fortress", "garden"]):
        constraints["simulator_compatibility"] = "ignition"
    elif any(kw in prompt_lower for kw in ["gazebo classic", "gazebo 9", "gazebo 11"]):
        constraints["simulator_compatibility"] = "gazebo_classic"

    # Semantic feature extraction
    feature_patterns = {
        "house": ["house", "home", "residential", "domestic"],
        "office": ["office", "workplace", "corporate", "desk"],
        "warehouse": ["warehouse", "storage", "logistics", "shelves", "racks"],
        "hospital": ["hospital", "medical", "healthcare", "clinic"],
        "store": ["store", "shop", "retail", "bookstore", "grocery"],
        "cafe": ["cafe", "restaurant", "coffee", "food service"],
        "factory": ["factory", "manufacturing", "industrial", "assembly", "production"],
        "city": ["city", "urban", "street", "road", "traffic", "intersection"],
        "forest": ["forest", "woods", "trees", "nature", "vegetation"],
        "agriculture": ["farm", "agriculture", "field", "crops", "rural"],
        "tunnel": ["tunnel", "underground", "cave", "mining", "subterranean"],
        "parking": ["parking", "lot", "garage"],
        "racing": ["racing", "track", "circuit", "driving"],
        "space": ["lunar", "mars", "moon", "space", "planetary", "rover"],
        "maze": ["maze", "labyrinth", "obstacles"],
        "disaster": ["rubble", "disaster", "debris", "rescue", "emergency"],
    }

    for feature, keywords in feature_patterns.items():
        if any(kw in prompt_lower for kw in keywords):
            constraints["semantic_features"].append(feature)

    return constraints


# =============================================================================
# World Search and Filtering Tools
# =============================================================================

@tool
def search_world_database(
    query: str,
    environment_type: Optional[str] = None,
    scale: Optional[str] = None,
    terrain_type: Optional[str] = None,
    simulator: Optional[str] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Search the world database using both symbolic filters and text search.

    This implements the symbolic filtering + text search strategy:
    1. Apply hard filters (environment, scale, terrain, simulator)
    2. Then perform text-based semantic search on filtered results

    Args:
        query: Natural language search query
        environment_type: Filter by "indoor", "outdoor", or "mixed"
        scale: Filter by "small", "medium", or "large"
        terrain_type: Filter by "flat", "uneven", "heightmap", or "multi_level"
        simulator: Filter by "gazebo_classic", "ignition", or "both"
        limit: Maximum number of results to return

    Returns:
        List of world metadata dictionaries sorted by relevance
    """
    # Convert string filters to enums
    env_filter = None
    if environment_type:
        env_map = {
            "indoor": EnvironmentType.INDOOR,
            "outdoor": EnvironmentType.OUTDOOR,
            "mixed": EnvironmentType.MIXED,
        }
        env_filter = env_map.get(environment_type.lower())

    scale_filter = None
    if scale:
        scale_map = {
            "small": Scale.SMALL,
            "medium": Scale.MEDIUM,
            "large": Scale.LARGE,
        }
        scale_filter = scale_map.get(scale.lower())

    terrain_filter = None
    if terrain_type:
        terrain_map = {
            "flat": TerrainType.FLAT,
            "uneven": TerrainType.UNEVEN,
            "heightmap": TerrainType.HEIGHTMAP,
            "multi_level": TerrainType.MULTI_LEVEL,
        }
        terrain_filter = terrain_map.get(terrain_type.lower())

    sim_filter = None
    if simulator:
        sim_map = {
            "gazebo_classic": SimulatorCompatibility.GAZEBO_CLASSIC,
            "ignition": SimulatorCompatibility.IGNITION,
            "both": SimulatorCompatibility.BOTH,
        }
        sim_filter = sim_map.get(simulator.lower())

    # Apply symbolic filters first
    filtered_worlds = filter_worlds(
        environment_type=env_filter,
        scale=scale_filter,
        terrain_type=terrain_filter,
        simulator_compatibility=sim_filter,
    )

    # If we have filtered results, perform text search on them
    if filtered_worlds and query:
        query_lower = query.lower()
        query_terms = query_lower.split()

        scored_results = []
        for world in filtered_worlds:
            score = 0
            searchable_text = (
                world.name.lower() + " " +
                world.description.lower() + " " +
                " ".join(world.semantic_features).lower()
            )

            for term in query_terms:
                if term in world.name.lower():
                    score += 10
                if term in world.description.lower():
                    score += 5
                if any(term in f.lower() for f in world.semantic_features):
                    score += 8
                if term in searchable_text:
                    score += 1

            scored_results.append((world, score))

        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        filtered_worlds = [w for w, s in scored_results]

    # If no filtered results but we have a query, fall back to text search
    elif query and not filtered_worlds:
        filtered_worlds = search_worlds_by_text(query, limit=limit * 2)

    # Convert to dictionaries and limit
    results = []
    for world in filtered_worlds[:limit]:
        results.append(world.to_dict())

    return results


@tool
def get_world_details(world_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific world by its ID.

    Args:
        world_id: The unique identifier of the world

    Returns:
        Dictionary with complete world metadata
    """
    world = get_world_by_id(world_id)
    if world:
        return world.to_dict()
    return {"error": f"World not found: {world_id}"}


@tool
def list_available_worlds(
    environment_type: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, str]]:
    """
    List all available worlds in the database with basic info.

    Args:
        environment_type: Optional filter for "indoor", "outdoor", or "mixed"
        limit: Maximum number of results

    Returns:
        List of dictionaries with id, name, and description
    """
    worlds = get_world_database()

    if environment_type:
        env_map = {
            "indoor": EnvironmentType.INDOOR,
            "outdoor": EnvironmentType.OUTDOOR,
            "mixed": EnvironmentType.MIXED,
        }
        env_filter = env_map.get(environment_type.lower())
        if env_filter:
            worlds = [w for w in worlds if w.environment_type == env_filter]

    results = []
    for world in worlds[:limit]:
        results.append({
            "id": world.id,
            "name": world.name,
            "description": world.description[:100] + "..." if len(world.description) > 100 else world.description,
            "environment": world.environment_type.value,
            "scale": world.scale.value,
        })

    return results


# =============================================================================
# ROS Package Management Tools
# =============================================================================

@tool
def find_worlds_package(workspace_path: str) -> Dict[str, Any]:
    """
    Find an existing worlds package in the ROS workspace.

    Searches for packages named:
    - *_worlds
    - *_simulation
    - *_gazebo

    Args:
        workspace_path: Path to the ROS workspace

    Returns:
        Dictionary with package information or indication that none exists
    """
    result = {
        "found": False,
        "package_name": None,
        "package_path": None,
        "worlds_dir": None,
        "existing_worlds": [],
        "candidates_checked": [],
    }

    src_path = os.path.join(workspace_path, "src")
    if not os.path.exists(src_path):
        result["error"] = f"No src directory found in workspace: {workspace_path}"
        return result

    # Search patterns for world packages
    patterns = ["_worlds", "_simulation", "_gazebo", "_sim"]

    for root, dirs, files in os.walk(src_path):
        if "package.xml" in files:
            pkg_name = os.path.basename(root)
            result["candidates_checked"].append(pkg_name)

            # Check if package name matches any pattern
            if any(pattern in pkg_name.lower() for pattern in patterns):
                # Check if it has a worlds directory
                worlds_dir = os.path.join(root, "worlds")

                result["found"] = True
                result["package_name"] = pkg_name
                result["package_path"] = root

                if os.path.exists(worlds_dir):
                    result["worlds_dir"] = worlds_dir
                    # List existing world files
                    for f in os.listdir(worlds_dir):
                        if f.endswith((".world", ".sdf")):
                            result["existing_worlds"].append(f)
                else:
                    result["worlds_dir"] = worlds_dir  # Will need to be created

                return result

    return result


@tool
def create_worlds_package(
    workspace_path: str,
    package_name: str = "my_worlds",
    description: str = "Gazebo world files for simulation",
    maintainer: str = "user",
    maintainer_email: str = "user@example.com"
) -> Dict[str, Any]:
    """
    Create a new ROS2 package for storing world files.

    Creates a proper ament_cmake package structure:
    ```
    my_worlds/
    ├── package.xml
    ├── CMakeLists.txt
    └── worlds/
        └── (world files go here)
    ```

    The package is configured to:
    - Install world files to share/<package_name>/worlds/
    - Include a launch file for easy world loading
    - Support both .world and .sdf formats

    Args:
        workspace_path: Path to the ROS workspace
        package_name: Name for the package (default: "my_worlds")
        description: Package description
        maintainer: Package maintainer name
        maintainer_email: Maintainer email

    Returns:
        Dictionary with creation status and paths
    """
    result = {
        "success": False,
        "package_path": None,
        "files_created": [],
        "errors": [],
    }

    src_path = os.path.join(workspace_path, "src")
    package_path = os.path.join(src_path, package_name)

    try:
        # Create directory structure
        os.makedirs(package_path, exist_ok=True)
        os.makedirs(os.path.join(package_path, "worlds"), exist_ok=True)
        os.makedirs(os.path.join(package_path, "launch"), exist_ok=True)
        os.makedirs(os.path.join(package_path, "models"), exist_ok=True)

        result["package_path"] = package_path

        # Create package.xml (ament_cmake for world files)
        package_xml = f'''<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{package_name}</name>
  <version>0.0.1</version>
  <description>{description}</description>
  <maintainer email="{maintainer_email}">{maintainer}</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <exec_depend>gazebo_ros</exec_depend>
  <exec_depend>ros2launch</exec_depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
'''
        package_xml_path = os.path.join(package_path, "package.xml")
        with open(package_xml_path, 'w') as f:
            f.write(package_xml)
        result["files_created"].append(package_xml_path)

        # Create CMakeLists.txt
        cmake_content = f'''cmake_minimum_required(VERSION 3.8)
project({package_name})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

# Install world files
install(
  DIRECTORY worlds/
  DESTINATION share/${{PROJECT_NAME}}/worlds
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${{PROJECT_NAME}}/launch
)

# Install model files
install(
  DIRECTORY models/
  DESTINATION share/${{PROJECT_NAME}}/models
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
'''
        cmake_path = os.path.join(package_path, "CMakeLists.txt")
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        result["files_created"].append(cmake_path)

        # Create launch file template
        launch_content = f'''#!/usr/bin/env python3
"""
Launch file for Gazebo worlds.

Usage:
    ros2 launch {package_name} world.launch.py world:=<world_name>
"""

import os
from ament_index.python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('{package_name}')

    # Declare arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty',
        description='Name of the world file (without .world extension)'
    )

    # World file path
    world_file = PathJoinSubstitution([
        pkg_share,
        'worlds',
        [LaunchConfiguration('world'), '.world']
    ])

    # Include Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ]),
        launch_arguments={{'world': world_file}}.items()
    )

    return LaunchDescription([
        world_arg,
        gazebo_launch,
    ])
'''
        launch_path = os.path.join(package_path, "launch", "world.launch.py")
        with open(launch_path, 'w') as f:
            f.write(launch_content)
        result["files_created"].append(launch_path)

        # Create placeholder world file
        empty_world = '''<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sun light -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics settings -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
  </world>
</sdf>
'''
        empty_world_path = os.path.join(package_path, "worlds", "empty.world")
        with open(empty_world_path, 'w') as f:
            f.write(empty_world)
        result["files_created"].append(empty_world_path)

        result["success"] = True

    except Exception as e:
        result["errors"].append(str(e))

    return result


@tool
def download_world_file(
    world_id: str,
    workspace_path: str,
    target_package: Optional[str] = None,
    custom_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download a world file from its source repository and place it in the correct ROS package.

    This tool:
    1. Looks up the world in the database
    2. Finds or creates a suitable worlds package
    3. Downloads the world file
    4. Places it in <package>/worlds/<name>.world

    Args:
        world_id: ID of the world from the database
        workspace_path: Path to the ROS workspace
        target_package: Optional specific package to use (otherwise auto-detected)
        custom_name: Optional custom name for the world file

    Returns:
        Dictionary with download status and file paths
    """
    result = {
        "success": False,
        "world_file_path": None,
        "package_used": None,
        "launch_command": None,
        "errors": [],
        "warnings": [],
    }

    # Get world metadata
    world = get_world_by_id(world_id)
    if not world:
        result["errors"].append(f"World not found in database: {world_id}")
        return result

    # Determine world file name
    if custom_name:
        world_name = custom_name if custom_name.endswith('.world') else f"{custom_name}.world"
    else:
        # Extract name from the source path
        source_name = os.path.basename(world.world_file_path)
        world_name = source_name if source_name.endswith('.world') or source_name.endswith('.sdf') else f"{world.id}.world"

    # Find or determine target package
    if target_package:
        package_path = os.path.join(workspace_path, "src", target_package)
        if not os.path.exists(package_path):
            result["errors"].append(f"Specified package not found: {target_package}")
            return result
        result["package_used"] = target_package
    else:
        # Auto-detect worlds package
        pkg_info = find_worlds_package.invoke({"workspace_path": workspace_path})
        if pkg_info.get("found"):
            package_path = pkg_info["package_path"]
            result["package_used"] = pkg_info["package_name"]
        else:
            # Need to create a new package - use simple "my_worlds" name
            new_pkg_name = "my_worlds"

            create_result = create_worlds_package.invoke({
                "workspace_path": workspace_path,
                "package_name": new_pkg_name,
            })

            if not create_result.get("success"):
                result["errors"].extend(create_result.get("errors", ["Failed to create worlds package"]))
                return result

            package_path = create_result["package_path"]
            result["package_used"] = new_pkg_name
            result["warnings"].append(f"Created new worlds package: {new_pkg_name}")

    # Ensure worlds directory exists
    worlds_dir = os.path.join(package_path, "worlds")
    os.makedirs(worlds_dir, exist_ok=True)

    # Construct download URL
    # Convert GitHub repo URL to raw content URL
    source_url = world.source_url
    file_path = world.world_file_path

    if "github.com" in source_url:
        # Convert to raw.githubusercontent.com URL
        repo_parts = source_url.replace("https://github.com/", "").rstrip('/')
        raw_url = f"https://raw.githubusercontent.com/{repo_parts}/main/{file_path}"
        # Also try master branch as fallback
        raw_url_master = f"https://raw.githubusercontent.com/{repo_parts}/master/{file_path}"
    else:
        result["errors"].append(f"Unsupported source URL format: {source_url}")
        return result

    # Download the file
    target_path = os.path.join(worlds_dir, world_name)

    try:
        # Try main branch first
        try:
            urllib.request.urlretrieve(raw_url, target_path)
        except urllib.error.HTTPError:
            # Try master branch
            urllib.request.urlretrieve(raw_url_master, target_path)

        result["success"] = True
        result["world_file_path"] = target_path
        result["launch_command"] = f"ros2 launch {result['package_used']} world.launch.py world:={os.path.splitext(world_name)[0]}"

    except Exception as e:
        result["errors"].append(f"Failed to download world file: {str(e)}")
        result["warnings"].append(
            f"You may need to manually download from: {source_url}"
        )

        # Create a placeholder with instructions
        placeholder_content = f'''<?xml version="1.0" ?>
<!--
  World: {world.name}
  Source: {world.source_url}
  File: {world.world_file_path}

  This is a placeholder file. The actual world file could not be downloaded automatically.

  To get the actual world file:
  1. Clone the repository: git clone {world.source_url}
  2. Copy the world file: {world.world_file_path}
  3. Place it here: {target_path}
-->
<sdf version="1.6">
  <world name="{world.id}">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
'''
        with open(target_path, 'w') as f:
            f.write(placeholder_content)

        result["world_file_path"] = target_path
        result["warnings"].append(f"Created placeholder world file at: {target_path}")

    return result


@tool
def write_world_file(
    workspace_path: str,
    world_content: str,
    world_name: str,
    target_package: Optional[str] = None
) -> Dict[str, Any]:
    """
    Write a world file directly to the appropriate ROS package.

    Use this when you have the world file content already (e.g., from a template
    or manual download).

    Args:
        workspace_path: Path to the ROS workspace
        world_content: The complete world file content (XML/SDF format)
        world_name: Name for the world file (e.g., "my_world" or "my_world.world")
        target_package: Optional specific package to use

    Returns:
        Dictionary with write status and file path
    """
    result = {
        "success": False,
        "world_file_path": None,
        "package_used": None,
        "launch_command": None,
        "errors": [],
    }

    # Ensure proper file extension
    if not world_name.endswith('.world') and not world_name.endswith('.sdf'):
        world_name = f"{world_name}.world"

    # Find or create target package
    if target_package:
        package_path = os.path.join(workspace_path, "src", target_package)
        if not os.path.exists(package_path):
            result["errors"].append(f"Package not found: {target_package}")
            return result
        result["package_used"] = target_package
    else:
        pkg_info = find_worlds_package.invoke({"workspace_path": workspace_path})
        if pkg_info.get("found"):
            package_path = pkg_info["package_path"]
            result["package_used"] = pkg_info["package_name"]
        else:
            # Create new package - use simple "my_worlds" name
            new_pkg_name = "my_worlds"

            create_result = create_worlds_package.invoke({
                "workspace_path": workspace_path,
                "package_name": new_pkg_name,
            })

            if not create_result.get("success"):
                result["errors"].extend(create_result.get("errors", ["Failed to create package"]))
                return result

            package_path = create_result["package_path"]
            result["package_used"] = new_pkg_name

    # Write world file
    worlds_dir = os.path.join(package_path, "worlds")
    os.makedirs(worlds_dir, exist_ok=True)

    world_path = os.path.join(worlds_dir, world_name)

    try:
        with open(world_path, 'w') as f:
            f.write(world_content)

        result["success"] = True
        result["world_file_path"] = world_path
        world_base = os.path.splitext(world_name)[0]
        result["launch_command"] = f"ros2 launch {result['package_used']} world.launch.py world:={world_base}"

    except Exception as e:
        result["errors"].append(f"Failed to write world file: {str(e)}")

    return result


@tool
def validate_world_file(world_file_path: str) -> Dict[str, Any]:
    """
    Validate a world file's structure and content.

    Checks:
    - File exists
    - Valid XML/SDF structure
    - Contains required elements (world, physics)
    - References valid models

    Args:
        world_file_path: Path to the world file

    Returns:
        Dictionary with validation results
    """
    result = {
        "is_valid": False,
        "file_exists": False,
        "is_valid_xml": False,
        "has_world_element": False,
        "has_physics": False,
        "has_ground": False,
        "has_light": False,
        "world_name": None,
        "models_referenced": [],
        "warnings": [],
        "errors": [],
    }

    if not os.path.exists(world_file_path):
        result["errors"].append(f"File not found: {world_file_path}")
        return result

    result["file_exists"] = True

    try:
        tree = ET.parse(world_file_path)
        root = tree.getroot()
        result["is_valid_xml"] = True

        # Check for world element
        world_elem = root.find(".//world")
        if world_elem is not None:
            result["has_world_element"] = True
            result["world_name"] = world_elem.get("name", "unnamed")

            # Check for physics
            if world_elem.find(".//physics") is not None:
                result["has_physics"] = True

            # Check for ground plane
            for include in world_elem.findall(".//include"):
                uri = include.find("uri")
                if uri is not None and uri.text:
                    model_name = uri.text.replace("model://", "")
                    result["models_referenced"].append(model_name)
                    if "ground" in model_name.lower():
                        result["has_ground"] = True

            # Check for light source
            if world_elem.find(".//light") is not None or \
               any("sun" in m.lower() for m in result["models_referenced"]):
                result["has_light"] = True

        else:
            result["errors"].append("No <world> element found in file")

        # Add warnings for missing recommended elements
        if not result["has_physics"]:
            result["warnings"].append("No physics configuration found")
        if not result["has_ground"]:
            result["warnings"].append("No ground plane detected")
        if not result["has_light"]:
            result["warnings"].append("No light source detected")

        # Overall validity
        result["is_valid"] = result["has_world_element"]

    except ET.ParseError as e:
        result["errors"].append(f"XML parse error: {str(e)}")
    except Exception as e:
        result["errors"].append(f"Validation error: {str(e)}")

    return result


@tool
def find_simulation_launch_files(workspace_path: str) -> Dict[str, Any]:
    """
    Find simulation launch files in the workspace that might need world path updates.

    Searches for launch files that:
    - Contain Gazebo/simulation-related keywords
    - Have world file path parameters
    - Are in simulation-related packages

    Args:
        workspace_path: Path to the ROS workspace

    Returns:
        Dictionary with list of launch files and their world-related configurations
    """
    result = {
        "launch_files": [],
        "simulation_packages": [],
        "errors": [],
    }

    src_path = os.path.join(workspace_path, "src")
    if not os.path.exists(src_path):
        result["errors"].append(f"No src directory found: {workspace_path}")
        return result

    # Keywords that indicate simulation launch files
    sim_keywords = ["gazebo", "simulation", "sim", "world", "spawn", "ignition"]
    world_patterns = [
        r'world[_\s]*[:=]',
        r'world_file',
        r'\.world',
        r'\.sdf',
        r'world_name',
        r'gazebo.*world',
    ]

    for root, dirs, files in os.walk(src_path):
        # Track simulation packages
        if "package.xml" in files:
            pkg_name = os.path.basename(root)
            if any(kw in pkg_name.lower() for kw in sim_keywords):
                result["simulation_packages"].append({
                    "name": pkg_name,
                    "path": root,
                })

        # Find launch files
        for f in files:
            if f.endswith((".launch.py", ".launch", ".launch.xml")):
                file_path = os.path.join(root, f)

                try:
                    with open(file_path, 'r') as fp:
                        content = fp.read()

                    # Check if it's simulation-related
                    is_sim_related = any(kw in content.lower() for kw in sim_keywords)

                    # Check for world-related patterns
                    has_world_config = any(
                        re.search(pattern, content, re.IGNORECASE)
                        for pattern in world_patterns
                    )

                    if is_sim_related or has_world_config:
                        # Extract world path if present
                        world_path_match = re.search(
                            r'["\']([^"\']*\.world)["\']|world\s*[:=]\s*["\']([^"\']+)["\']',
                            content
                        )
                        current_world = None
                        if world_path_match:
                            current_world = world_path_match.group(1) or world_path_match.group(2)

                        # Determine package
                        rel_path = os.path.relpath(file_path, src_path)
                        pkg_name = rel_path.split(os.sep)[0] if os.sep in rel_path else ""

                        result["launch_files"].append({
                            "name": f,
                            "path": file_path,
                            "package": pkg_name,
                            "is_simulation": is_sim_related,
                            "has_world_config": has_world_config,
                            "current_world": current_world,
                            "file_type": "python" if f.endswith(".py") else "xml",
                        })

                except Exception as e:
                    result["errors"].append(f"Error reading {file_path}: {str(e)}")

    return result


@tool
def update_simulation_launch_world(
    launch_file_path: str,
    worlds_package_name: str,
    world_file_name: str,
    backup: bool = True
) -> Dict[str, Any]:
    """
    Update a simulation launch file to use a new world from the worlds package.

    This tool modifies existing launch files to point to the newly created world file
    in the my_worlds (or similar) package. It handles both Python and XML launch files.

    Args:
        launch_file_path: Path to the launch file to update
        worlds_package_name: Name of the package containing the world (e.g., "my_worlds")
        world_file_name: Name of the world file (e.g., "office.world" or "office")
        backup: Whether to create a backup of the original file

    Returns:
        Dictionary with update status and details
    """
    result = {
        "success": False,
        "backup_path": None,
        "changes_made": [],
        "new_world_path_code": None,
        "errors": [],
    }

    if not os.path.exists(launch_file_path):
        result["errors"].append(f"Launch file not found: {launch_file_path}")
        return result

    # Ensure world file has proper extension
    if not world_file_name.endswith(('.world', '.sdf')):
        world_file_name = f"{world_file_name}.world"

    world_base_name = os.path.splitext(world_file_name)[0]

    try:
        with open(launch_file_path, 'r') as f:
            original_content = f.read()

        # Create backup if requested
        if backup:
            backup_path = f"{launch_file_path}.backup"
            with open(backup_path, 'w') as f:
                f.write(original_content)
            result["backup_path"] = backup_path

        new_content = original_content
        is_python = launch_file_path.endswith('.py')

        if is_python:
            # Handle Python launch files

            # Check if get_package_share_directory import exists
            if "get_package_share_directory" not in original_content:
                # Add the import
                import_line = "from ament_index.python.packages import get_package_share_directory\n"
                # Find where to insert (after other imports or at the beginning)
                import_match = re.search(r'^(import|from)\s+', original_content, re.MULTILINE)
                if import_match:
                    # Find the last import line
                    last_import = 0
                    for match in re.finditer(r'^(import|from)\s+.+$', original_content, re.MULTILINE):
                        last_import = match.end()
                    new_content = original_content[:last_import] + "\n" + import_line + original_content[last_import:]
                else:
                    new_content = import_line + original_content
                result["changes_made"].append("Added get_package_share_directory import")

            # Generate the new world path code
            world_path_code = f"""
    # World file path from {worlds_package_name}
    worlds_pkg_share = get_package_share_directory('{worlds_package_name}')
    world_file_path = os.path.join(worlds_pkg_share, 'worlds', '{world_file_name}')
"""
            result["new_world_path_code"] = world_path_code.strip()

            # Common patterns to replace
            replacements = [
                # Pattern: world = '...' or world='...'
                (r"world\s*=\s*['\"][^'\"]+\.world['\"]",
                 f"world = world_file_path"),
                # Pattern: 'world': '...'
                (r"['\"]world['\"]\s*:\s*['\"][^'\"]+\.world['\"]",
                 f"'world': world_file_path"),
                # Pattern: world_file = '...'
                (r"world_file\s*=\s*['\"][^'\"]+['\"]",
                 f"world_file = world_file_path"),
            ]

            for pattern, replacement in replacements:
                if re.search(pattern, new_content):
                    new_content = re.sub(pattern, replacement, new_content)
                    result["changes_made"].append(f"Updated world path pattern: {pattern[:30]}...")

            # Check if we need to add the world path code
            if "world_file_path" in new_content and f"worlds_pkg_share = get_package_share_directory('{worlds_package_name}')" not in new_content:
                # Find generate_launch_description function
                func_match = re.search(r'def\s+generate_launch_description\s*\(\s*\)\s*:', new_content)
                if func_match:
                    # Insert world path code at the beginning of the function
                    insert_pos = func_match.end()
                    new_content = new_content[:insert_pos] + world_path_code + new_content[insert_pos:]
                    result["changes_made"].append("Inserted world path code in generate_launch_description")

                    # Also ensure os import exists
                    if "import os" not in new_content:
                        new_content = "import os\n" + new_content
                        result["changes_made"].append("Added os import")

        else:
            # Handle XML launch files
            # Replace world file paths in XML
            xml_patterns = [
                # Pattern: <arg name="world" default="..."/>
                (r'(<arg\s+name=["\']world["\']\s+default=)["\'][^"\']+\.world["\']',
                 f'\\1"$(find {worlds_package_name})/worlds/{world_file_name}"'),
                # Pattern: world:="..."
                (r'world:=["\'][^"\']+\.world["\']',
                 f'world:="$(find {worlds_package_name})/worlds/{world_file_name}"'),
            ]

            for pattern, replacement in xml_patterns:
                if re.search(pattern, new_content):
                    new_content = re.sub(pattern, replacement, new_content)
                    result["changes_made"].append(f"Updated XML world path")

        # Write the modified content
        if new_content != original_content:
            with open(launch_file_path, 'w') as f:
                f.write(new_content)
            result["success"] = True
        else:
            result["errors"].append("No world path patterns found to update. Manual modification may be required.")
            # Provide guidance
            result["manual_instructions"] = f"""
To manually update the launch file, add:

For Python launch files:
1. Add import: from ament_index.python.packages import get_package_share_directory
2. Add import: import os
3. In generate_launch_description(), add:
   worlds_pkg_share = get_package_share_directory('{worlds_package_name}')
   world_file_path = os.path.join(worlds_pkg_share, 'worlds', '{world_file_name}')
4. Use world_file_path where the world is specified

For XML launch files:
   Use: $(find {worlds_package_name})/worlds/{world_file_name}
"""
            result["success"] = True  # Consider it success with manual instructions

    except Exception as e:
        result["errors"].append(f"Error updating launch file: {str(e)}")

    return result


@tool
def generate_world_launch_snippet(
    package_name: str,
    world_name: str,
    use_sim_time: bool = True
) -> Dict[str, str]:
    """
    Generate a launch file snippet for launching a Gazebo world.

    Args:
        package_name: Name of the ROS package containing the world
        world_name: Name of the world file (without .world extension)
        use_sim_time: Whether to use simulation time

    Returns:
        Dictionary with launch file code and usage instructions
    """
    launch_code = f'''#!/usr/bin/env python3
"""
Launch Gazebo with {world_name} world.

Generated by Simbo World Design Agent.
"""

import os
from ament_index.python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Get package directories
    pkg_share = get_package_share_directory('{package_name}')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')

    # World file path
    world_file = os.path.join(pkg_share, 'worlds', '{world_name}.world')

    # Set Gazebo model path to include package models
    gazebo_models_path = os.path.join(pkg_share, 'models')

    return LaunchDescription([
        # Set environment variables for Gazebo
        SetEnvironmentVariable(
            name='GAZEBO_MODEL_PATH',
            value=[gazebo_models_path, ':', os.environ.get('GAZEBO_MODEL_PATH', '')]
        ),

        # Use simulation time
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='{"true" if use_sim_time else "false"}',
            description='Use simulation (Gazebo) clock if true'
        ),

        # Launch Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')
            ),
            launch_arguments={{
                'world': world_file,
                'verbose': 'true',
            }}.items()
        ),
    ])
'''

    usage = f'''# Launch the world:
ros2 launch {package_name} {world_name}_world.launch.py

# Or with the generic launcher:
ros2 launch {package_name} world.launch.py world:={world_name}

# To spawn a robot in this world, add to your robot launch file:
# from ament_index.python.packages import get_package_share_directory
# world_path = os.path.join(get_package_share_directory('{package_name}'), 'worlds', '{world_name}.world')
'''

    return {
        "launch_code": launch_code,
        "usage_instructions": usage,
        "launch_file_name": f"{world_name}_world.launch.py",
    }
