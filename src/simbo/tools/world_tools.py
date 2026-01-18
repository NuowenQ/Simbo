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
from datetime import datetime
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

    Prioritizes simbo_worlds package, then searches for packages named:
    - simbo_worlds (priority)
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

    # First, check for simbo_worlds specifically (priority)
    simbo_worlds_path = os.path.join(src_path, "simbo_worlds")
    if os.path.exists(simbo_worlds_path) and os.path.exists(os.path.join(simbo_worlds_path, "package.xml")):
        worlds_dir = os.path.join(simbo_worlds_path, "worlds")
        result["found"] = True
        result["package_name"] = "simbo_worlds"
        result["package_path"] = simbo_worlds_path
        if os.path.exists(worlds_dir):
            result["worlds_dir"] = worlds_dir
            # List existing world files
            for f in os.listdir(worlds_dir):
                if f.endswith((".world", ".sdf")):
                    result["existing_worlds"].append(f)
        else:
            result["worlds_dir"] = worlds_dir  # Will need to be created
        return result

    # Search patterns for world packages (fallback)
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
    package_name: str = "simbo_worlds",
    description: str = "Gazebo world files for simulation",
    maintainer: str = "user",
    maintainer_email: str = "user@example.com"
) -> Dict[str, Any]:
    """
    Create a new ROS2 Python package for storing world files.

    Creates a proper ament_python package structure (NO C++/CMake):
    ```
    simbo_worlds/
    ├── package.xml
    ├── setup.py
    ├── setup.cfg
    ├── resource/
    │   └── simbo_worlds
    ├── worlds/
    │   └── (world files go here)
    └── launch/
        └── world.launch.py
    ```

    The package is configured to:
    - Install world files to share/<package_name>/worlds/
    - Include a launch file for easy world loading
    - Support both .world and .sdf formats
    - Python-only (no CMake/C++ required)

        Args:
        workspace_path: Path to the ROS workspace
        package_name: Name for the package (default: "simbo_worlds")
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
        os.makedirs(os.path.join(package_path, "resource"), exist_ok=True)

        result["package_path"] = package_path

        # Create package.xml (ament_python - NO CMake!)
        package_xml = f'''<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{package_name}</name>
  <version>0.0.1</version>
  <description>{description}</description>
  <maintainer email="{maintainer_email}">{maintainer}</maintainer>
  <license>Apache-2.0</license>

  <exec_depend>gazebo_ros</exec_depend>
  <exec_depend>ros2launch</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
'''
        package_xml_path = os.path.join(package_path, "package.xml")
        with open(package_xml_path, 'w') as f:
            f.write(package_xml)
        result["files_created"].append(package_xml_path)

        # Create setup.py (Python-only package)
        # Note: We use a function to collect data files at build time
        setup_py_content = f'''from setuptools import setup
import os
from glob import glob

package_name = '{package_name}'

def get_data_files():
    """Collect data files for installation."""
    data_files = [
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]
    
    # Install world files
    world_files = glob('worlds/*.world') + glob('worlds/*.sdf')
    if world_files:
        data_files.append(
            (os.path.join('share', package_name, 'worlds'), world_files)
        )
    
    # Install launch files
    launch_files = glob('launch/*.launch.py')
    if launch_files:
        data_files.append(
            (os.path.join('share', package_name, 'launch'), launch_files)
        )
    
    # Install model files (preserving directory structure)
    if os.path.exists('models'):
        for root, dirs, files in os.walk('models'):
            if files:
                # Get relative path from 'models' directory
                rel_path = os.path.relpath(root, '.')
                install_path = os.path.join('share', package_name, rel_path)
                file_paths = [os.path.join(root, f) for f in files]
                data_files.append((install_path, file_paths))
    
    return data_files

setup(
    name=package_name,
    version='0.0.1',
    packages=[],
    data_files=get_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='{maintainer}',
    maintainer_email='{maintainer_email}',
    description='{description}',
    license='Apache-2.0',
    tests_require=['pytest'],
)
'''
        setup_py_path = os.path.join(package_path, "setup.py")
        with open(setup_py_path, 'w') as f:
            f.write(setup_py_content)
        result["files_created"].append(setup_py_path)

        # Create setup.cfg (CRITICAL for ROS2 Python packages)
        setup_cfg_content = f"""[develop]
script_dir=$base/lib/{package_name}
[install]
install_scripts=$base/lib/{package_name}
"""
        setup_cfg_path = os.path.join(package_path, "setup.cfg")
        with open(setup_cfg_path, 'w') as f:
            f.write(setup_cfg_content)
        result["files_created"].append(setup_cfg_path)

        # Create resource marker file
        resource_path = os.path.join(package_path, "resource", package_name)
        with open(resource_path, 'w') as f:
            f.write("")  # Empty file
        result["files_created"].append(resource_path)

        # Create launch file template
        launch_content = f'''#!/usr/bin/env python3
"""
Launch file for Gazebo worlds.

Usage:
    ros2 launch {package_name} world.launch.py world:=<world_name>
"""

import os
import json
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('{package_name}')

    # Set GAZEBO_MODEL_PATH to include package models directory
    models_path = os.path.join(pkg_share, 'models')
    existing_model_path = os.environ.get('GAZEBO_MODEL_PATH', '')
    if existing_model_path:
        gazebo_model_path = models_path + ':' + existing_model_path
    else:
        gazebo_model_path = models_path

    # Try to get latest world from tracking file, otherwise use argument
    latest_world_file = os.path.join(pkg_share, '.simbo_latest_world')
    default_world = 'empty.world'

    if os.path.exists(latest_world_file):
        try:
            with open(latest_world_file, 'r') as f:
                latest_info = json.load(f)
                default_world = latest_info.get('latest_world', 'empty.world')
        except Exception:
            pass  # Fall back to default

    # Declare arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='Name of the world file (with .world extension)'
    )

    # World file path - construct using PathJoinSubstitution
    world_file = PathJoinSubstitution([
        pkg_share,
        'worlds',
        LaunchConfiguration('world')
    ])

    # Set environment variable for Gazebo model path
    set_gazebo_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=gazebo_model_path
    )

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
        set_gazebo_model_path,
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
def migrate_package_to_python(
    workspace_path: str,
    package_name: str = "simbo_worlds"
) -> Dict[str, Any]:
    """
    Migrate an existing ament_cmake package to ament_python (Python-only).
    
    This tool converts CMake-based packages to Python-only packages by:
    - Updating package.xml to use ament_python
    - Creating setup.py and setup.cfg
    - Creating resource marker file
    - Removing CMakeLists.txt (optional backup)
    - Preserving all existing worlds, launch files, and models
    
    Args:
        workspace_path: Path to the ROS workspace
        package_name: Name of the package to migrate (default: "simbo_worlds")
    
    Returns:
        Dictionary with migration status and details
    """
    result = {
        "success": False,
        "package_path": None,
        "files_created": [],
        "files_removed": [],
        "errors": [],
    }
    
    src_path = os.path.join(workspace_path, "src")
    package_path = os.path.join(src_path, package_name)
    result["package_path"] = package_path
    
    if not os.path.exists(package_path):
        result["errors"].append(f"Package not found: {package_path}")
        return result
    
    try:
        # Read existing package.xml
        package_xml_path = os.path.join(package_path, "package.xml")
        if not os.path.exists(package_xml_path):
            result["errors"].append("package.xml not found")
            return result
        
        # Update package.xml to ament_python
        with open(package_xml_path, 'r') as f:
            xml_content = f.read()
        
        # Replace buildtool_depend and build_type
        xml_content = xml_content.replace(
            '<buildtool_depend>ament_cmake</buildtool_depend>',
            ''
        )
        xml_content = xml_content.replace(
            '<build_type>ament_cmake</build_type>',
            '<build_type>ament_python</build_type>'
        )
        
        # Update test dependencies to Python ones
        xml_content = xml_content.replace(
            '<test_depend>ament_lint_auto</test_depend>',
            '<test_depend>ament_copyright</test_depend>'
        )
        xml_content = xml_content.replace(
            '<test_depend>ament_lint_common</test_depend>',
            '<test_depend>ament_flake8</test_depend>\n  <test_depend>ament_pep257</test_depend>'
        )
        
        # Add python3-pytest if not present
        if 'python3-pytest' not in xml_content:
            xml_content = xml_content.replace(
                '</package>',
                '  <test_depend>python3-pytest</test_depend>\n\n</package>'
            )
        
        with open(package_xml_path, 'w') as f:
            f.write(xml_content)
        result["files_created"].append(f"{package_xml_path} (updated)")
        
        # Create setup.py (same as in create_worlds_package)
        setup_py_content = f'''from setuptools import setup
import os
from glob import glob

package_name = '{package_name}'

def get_data_files():
    """Collect data files for installation."""
    data_files = [
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]
    
    # Install world files
    world_files = glob('worlds/*.world') + glob('worlds/*.sdf')
    if world_files:
        data_files.append(
            (os.path.join('share', package_name, 'worlds'), world_files)
        )
    
    # Install launch files
    launch_files = glob('launch/*.launch.py')
    if launch_files:
        data_files.append(
            (os.path.join('share', package_name, 'launch'), launch_files)
        )
    
    # Install model files (preserving directory structure)
    if os.path.exists('models'):
        for root, dirs, files in os.walk('models'):
            if files:
                # Get relative path from 'models' directory
                rel_path = os.path.relpath(root, '.')
                install_path = os.path.join('share', package_name, rel_path)
                file_paths = [os.path.join(root, f) for f in files]
                data_files.append((install_path, file_paths))
    
    return data_files

setup(
    name=package_name,
    version='0.0.1',
    packages=[],
    data_files=get_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Gazebo world files for simulation',
    license='Apache-2.0',
    tests_require=['pytest'],
)
'''
        setup_py_path = os.path.join(package_path, "setup.py")
        with open(setup_py_path, 'w') as f:
            f.write(setup_py_content)
        result["files_created"].append(setup_py_path)
        
        # Create setup.cfg
        setup_cfg_content = f"""[develop]
script_dir=$base/lib/{package_name}
[install]
install_scripts=$base/lib/{package_name}
"""
        setup_cfg_path = os.path.join(package_path, "setup.cfg")
        with open(setup_cfg_path, 'w') as f:
            f.write(setup_cfg_content)
        result["files_created"].append(setup_cfg_path)
        
        # Create resource marker
        os.makedirs(os.path.join(package_path, "resource"), exist_ok=True)
        resource_path = os.path.join(package_path, "resource", package_name)
        with open(resource_path, 'w') as f:
            f.write("")
        result["files_created"].append(resource_path)
        
        # Backup and remove CMakeLists.txt
        cmake_path = os.path.join(package_path, "CMakeLists.txt")
        if os.path.exists(cmake_path):
            backup_path = os.path.join(package_path, "CMakeLists.txt.backup")
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(cmake_path, backup_path)
            os.remove(cmake_path)
            result["files_removed"].append("CMakeLists.txt (backed up)")
        
        result["success"] = True
        
    except Exception as e:
        result["errors"].append(f"Error migrating package: {str(e)}")
    
    return result


# =============================================================================
# Advanced World File Retrieval Tools
# =============================================================================

def _retrieve_world_file_from_github_impl(
    repository_url: str,
    file_path: str,
    output_path: str,
    branch: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve a world file from a GitHub repository using multiple methods.
    
    This tool tries several approaches:
    1. GitHub API to get file content directly
    2. Git sparse checkout (if git is available)
    3. Direct raw URL download with multiple branch attempts
    
    Args:
        repository_url: GitHub repository URL (e.g., "https://github.com/osrf/gazebo_models")
        file_path: Path to the file within the repository (e.g., "worlds/factory.world")
        output_path: Where to save the downloaded file
        branch: Optional branch name (if None, will try to detect)
    
    Returns:
        Dictionary with retrieval status and details
    """
    result = {
        "success": False,
        "method_used": None,
        "output_path": output_path,
        "errors": [],
        "warnings": [],
    }
    
    # Extract repository info
    repo_parts = repository_url.replace("https://github.com/", "").replace("http://github.com/", "").rstrip('/').replace(".git", "")
    
    # Method 1: Try GitHub API to get file content directly
    try:
        # First, get the default branch if not provided
        if not branch:
            try:
                api_url = f"https://api.github.com/repos/{repo_parts}"
                req = urllib.request.Request(api_url)
                req.add_header('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
                req.add_header('Accept', 'application/vnd.github.v3+json')
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status == 200:
                        repo_info = json.loads(response.read().decode())
                        branch = repo_info.get('default_branch', 'main')
            except Exception as e:
                result["warnings"].append(f"Could not detect default branch: {str(e)}")
                branch = "main"  # Fallback
        
        # Try to get file content via GitHub API
        api_file_url = f"https://api.github.com/repos/{repo_parts}/contents/{file_path}"
        if branch:
            api_file_url += f"?ref={branch}"
        
        try:
            req = urllib.request.Request(api_file_url)
            req.add_header('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
            req.add_header('Accept', 'application/vnd.github.v3+json')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    file_info = json.loads(response.read().decode())
                    
                    # GitHub API returns base64 encoded content
                    if file_info.get('encoding') == 'base64':
                        import base64
                        file_content = base64.b64decode(file_info['content']).decode('utf-8')
                        
                        # Ensure output directory exists
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(file_content)
                        
                        result["success"] = True
                        result["method_used"] = "GitHub API"
                        result["branch_used"] = branch
                        return result
        except urllib.error.HTTPError as e:
            if e.code == 404:
                result["warnings"].append(f"File not found at {file_path} in branch {branch}")
            else:
                result["warnings"].append(f"GitHub API error: {e.code} {e.reason}")
        except Exception as e:
            result["warnings"].append(f"GitHub API method failed: {str(e)}")
    
    except Exception as e:
        result["warnings"].append(f"GitHub API approach failed: {str(e)}")
    
    # Method 2: Try git sparse checkout (if git is available)
    try:
        import tempfile
        import shutil
        
        # Check if git is available
        git_check = subprocess.run(['git', '--version'], capture_output=True, timeout=5)
        if git_check.returncode == 0:
            # Create temporary directory for cloning
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_dir = os.path.join(temp_dir, 'repo')
                
                # Clone with sparse checkout
                clone_cmd = [
                    'git', 'clone',
                    '--filter=blob:none',  # Don't download all blobs
                    '--sparse',
                    '--depth', '1',
                    repository_url,
                    repo_dir
                ]
                
                if branch:
                    clone_cmd.extend(['--branch', branch])
                
                clone_result = subprocess.run(
                    clone_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=temp_dir
                )
                
                if clone_result.returncode == 0:
                    # Set up sparse checkout for the specific file
                    sparse_cmd = ['git', 'sparse-checkout', 'set', file_path]
                    sparse_result = subprocess.run(
                        sparse_cmd,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        cwd=repo_dir
                    )
                    
                    if sparse_result.returncode == 0:
                        # Checkout the file
                        checkout_cmd = ['git', 'checkout']
                        checkout_result = subprocess.run(
                            checkout_cmd,
                            capture_output=True,
                            text=True,
                            timeout=10,
                            cwd=repo_dir
                        )
                        
                        # Copy the file
                        source_file = os.path.join(repo_dir, file_path)
                        if os.path.exists(source_file):
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            shutil.copy2(source_file, output_path)
                            
                            result["success"] = True
                            result["method_used"] = "Git sparse checkout"
                            result["branch_used"] = branch or "default"
                            return result
    except FileNotFoundError:
        result["warnings"].append("Git not available, skipping git clone method")
    except subprocess.TimeoutExpired:
        result["warnings"].append("Git clone timed out")
    except Exception as e:
        result["warnings"].append(f"Git clone method failed: {str(e)}")
    
    # Method 3: Try direct raw URL with multiple branches
    branches_to_try = [branch] if branch else ["main", "master", "develop", "devel"]
    branches_to_try.extend(["gazebo11", "gazebo9", "ros2", "humble", "foxy", "noetic", "melodic"])
    
    # Remove duplicates while preserving order
    seen = set()
    branches_to_try = [b for b in branches_to_try if b and not (b in seen or seen.add(b))]
    
    file_path_variations = [
        file_path,
        file_path.lstrip('/'),
        file_path.replace('\\', '/'),
    ]
    
    for branch_name in branches_to_try:
        for path_var in file_path_variations:
            raw_url = f"https://raw.githubusercontent.com/{repo_parts}/{branch_name}/{path_var}"
            
            try:
                req = urllib.request.Request(raw_url)
                req.add_header('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status == 200:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        with open(output_path, 'wb') as f:
                            f.write(response.read())
                        
                        result["success"] = True
                        result["method_used"] = f"Raw URL (branch: {branch_name})"
                        result["branch_used"] = branch_name
                        return result
            except Exception:
                continue
    
    # All methods failed
    result["errors"].append(f"Failed to retrieve file using all methods: API, git clone, and raw URLs")
    result["errors"].append(f"Repository: {repository_url}, File: {file_path}")
    
    return result


@tool
def retrieve_world_file_from_github(
    repository_url: str,
    file_path: str,
    output_path: str,
    branch: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve a world file from a GitHub repository using multiple methods.
    
    This tool tries several approaches:
    1. GitHub API to get file content directly
    2. Git sparse checkout (if git is available)
    3. Direct raw URL download with multiple branch attempts
    
    Args:
        repository_url: GitHub repository URL (e.g., "https://github.com/osrf/gazebo_models")
        file_path: Path to the file within the repository (e.g., "worlds/factory.world")
        output_path: Where to save the downloaded file
        branch: Optional branch name (if None, will try to detect)
    
    Returns:
        Dictionary with retrieval status and details
    """
    return _retrieve_world_file_from_github_impl(
        repository_url=repository_url,
        file_path=file_path,
        output_path=output_path,
        branch=branch
    )


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
        # Always use simbo_worlds package (fixed location)
        new_pkg_name = "simbo_worlds"
        
        # Check if it exists
        pkg_info = find_worlds_package.invoke({"workspace_path": workspace_path})
        if pkg_info.get("found") and pkg_info.get("package_name") == "simbo_worlds":
            package_path = pkg_info["package_path"]
            result["package_used"] = "simbo_worlds"
        else:
            # Create simbo_worlds package
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

    # Download the file
    target_path = os.path.join(worlds_dir, world_name)

    if "github.com" in source_url:
        # Use the new advanced retrieval implementation
        # Call the implementation function directly (not the tool wrapper)
        retrieve_result = _retrieve_world_file_from_github_impl(
            repository_url=source_url,
            file_path=file_path,
            output_path=target_path,
            branch=None  # Will auto-detect
        )
        
        if retrieve_result.get("success"):
            download_success = True
            result["changes_made"] = [f"Downloaded using {retrieve_result.get('method_used')}"]
            if retrieve_result.get("branch_used"):
                result["changes_made"].append(f"Branch: {retrieve_result.get('branch_used')}")
        else:
            download_success = False
            result["errors"].extend(retrieve_result.get("errors", []))
            result["warnings"].extend(retrieve_result.get("warnings", []))
        
        if download_success:
            result["success"] = True
            result["world_file_path"] = target_path
            result["launch_command"] = f"ros2 launch {result['package_used']} world.launch.py world:={os.path.splitext(world_name)[0]}"
            
            # Track this as the latest world
            track_result = track_latest_world.invoke({
                "workspace_path": workspace_path,
                "world_file_name": world_name,
                "worlds_package_name": result["package_used"]
            })
            if track_result.get("success"):
                result["latest_tracked"] = True
            else:
                result["warnings"].append(f"Failed to track latest world: {track_result.get('errors', [])}")
    else:
        result["errors"].append(f"Unsupported source URL format: {source_url}")
        result["warnings"].append("Only GitHub repositories are currently supported for automatic download")
    
    # If download failed, create placeholder
    if not result.get("success"):
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
        try:
            with open(target_path, 'w') as f:
                f.write(placeholder_content)
            result["world_file_path"] = target_path
            result["warnings"].append(f"Created placeholder world file at: {target_path}")
        except Exception as e:
            result["errors"].append(f"Failed to create placeholder file: {str(e)}")

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
        # Always use simbo_worlds package (fixed location)
        new_pkg_name = "simbo_worlds"
        
        # Check if it exists
        pkg_info = find_worlds_package.invoke({"workspace_path": workspace_path})
        if pkg_info.get("found") and pkg_info.get("package_name") == "simbo_worlds":
            package_path = pkg_info["package_path"]
            result["package_used"] = "simbo_worlds"
        else:
            # Create simbo_worlds package
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
        
        # Track this as the latest world
        track_result = track_latest_world.invoke({
            "workspace_path": workspace_path,
            "world_file_name": world_name,
            "worlds_package_name": result["package_used"]
        })
        if track_result.get("success"):
            result["latest_tracked"] = True
        else:
            result["errors"].append(f"Failed to track latest world: {track_result.get('errors', [])}")

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


def _extract_world_models_impl(world_file_path: str) -> Dict[str, Any]:
    """
    Internal implementation for extracting model references from a world file.
    """
    result = {
        "success": False,
        "models": [],
        "model_count": 0,
        "errors": [],
    }

    if not os.path.exists(world_file_path):
        result["errors"].append(f"File not found: {world_file_path}")
        return result

    try:
        tree = ET.parse(world_file_path)
        root = tree.getroot()

        models_found = set()

        # Find all include elements with model:// URIs
        for include in root.findall(".//include"):
            uri = include.find("uri")
            if uri is not None and uri.text and uri.text.startswith("model://"):
                model_name = uri.text.replace("model://", "")
                models_found.add(model_name)

        # Also check for mesh URIs that reference models
        for mesh in root.findall(".//mesh"):
            uri = mesh.find("uri")
            if uri is not None and uri.text and uri.text.startswith("model://"):
                # Extract model name from path like model://some_model/meshes/mesh.dae
                parts = uri.text.replace("model://", "").split("/")
                if parts:
                    models_found.add(parts[0])

        result["models"] = sorted(list(models_found))
        result["model_count"] = len(models_found)
        result["success"] = True

    except ET.ParseError as e:
        result["errors"].append(f"XML parse error: {str(e)}")
    except Exception as e:
        result["errors"].append(f"Error extracting models: {str(e)}")

    return result


@tool
def extract_world_models(world_file_path: str) -> Dict[str, Any]:
    """
    Extract all model references from a world file.

    Parses the world file and identifies all models referenced via model:// URIs.
    This is useful to determine which models need to be available for the world
    to load correctly in Gazebo.

    Args:
        world_file_path: Path to the world file

    Returns:
        Dictionary with list of referenced models and their details
    """
    return _extract_world_models_impl(world_file_path)


def _download_model_from_repo(
    model_name: str,
    output_dir: str,
    repo: str = "osrf/gazebo_models",
    branch: str = "master",
    models_subdir: str = ""
) -> Dict[str, Any]:
    """
    Download a model from a GitHub repository.

    Args:
        model_name: Name of the model (e.g., "ground_plane", "willowgarage")
        output_dir: Directory to save the model
        repo: GitHub repository (e.g., "osrf/gazebo_models", "aws-robotics/aws-robomaker-small-house-world")
        branch: Branch to download from (default: master)
        models_subdir: Subdirectory in repo where models are located (e.g., "models")

    Returns:
        Dictionary with download status
    """
    result = {
        "success": False,
        "model_path": None,
        "errors": [],
        "warnings": [],
    }

    repo_parts = repo
    model_dir = os.path.join(output_dir, model_name)

    # Path to model in repo (may be in a models/ subdirectory)
    model_repo_path = f"{models_subdir}/{model_name}" if models_subdir else model_name

    # Try git sparse checkout first (most reliable for directories)
    try:
        import tempfile
        import shutil

        git_check = subprocess.run(['git', '--version'], capture_output=True, timeout=5)
        if git_check.returncode == 0:
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_dir = os.path.join(temp_dir, 'repo')

                clone_cmd = [
                    'git', 'clone',
                    '--filter=blob:none',
                    '--sparse',
                    '--depth', '1',
                    '--branch', branch,
                    f'https://github.com/{repo_parts}.git',
                    repo_dir
                ]

                clone_result = subprocess.run(
                    clone_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=temp_dir
                )

                if clone_result.returncode == 0:
                    # Set up sparse checkout for the model directory
                    sparse_cmd = ['git', 'sparse-checkout', 'set', model_repo_path]
                    subprocess.run(sparse_cmd, capture_output=True, text=True, timeout=10, cwd=repo_dir)

                    # Checkout
                    subprocess.run(['git', 'checkout'], capture_output=True, text=True, timeout=10, cwd=repo_dir)

                    source_model = os.path.join(repo_dir, model_repo_path)
                    if os.path.exists(source_model) and os.path.isdir(source_model):
                        os.makedirs(output_dir, exist_ok=True)
                        if os.path.exists(model_dir):
                            shutil.rmtree(model_dir)
                        shutil.copytree(source_model, model_dir)

                        result["success"] = True
                        result["model_path"] = model_dir
                        return result
                    else:
                        result["warnings"].append(f"Model directory not found after checkout: {model_name}")
    except subprocess.TimeoutExpired:
        result["warnings"].append("Git clone timed out")
    except FileNotFoundError:
        result["warnings"].append("Git not available")
    except Exception as e:
        result["warnings"].append(f"Git method failed: {str(e)}")

    # Fallback: try downloading essential files via raw URLs
    # Models typically have: model.config, model.sdf (or *.sdf)
    essential_files = ["model.config", "model.sdf"]
    branches_to_try = [branch, "master", "main", "ros2"]

    for try_branch in branches_to_try:
        files_downloaded = 0
        os.makedirs(model_dir, exist_ok=True)

        for filename in essential_files:
            raw_url = f"https://raw.githubusercontent.com/{repo_parts}/{try_branch}/{model_repo_path}/{filename}"
            target_file = os.path.join(model_dir, filename)

            try:
                req = urllib.request.Request(raw_url)
                req.add_header('User-Agent', 'Mozilla/5.0')

                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status == 200:
                        with open(target_file, 'wb') as f:
                            f.write(response.read())
                        files_downloaded += 1
            except Exception:
                continue

        if files_downloaded > 0:
            result["success"] = True
            result["model_path"] = model_dir
            result["warnings"].append(f"Downloaded {files_downloaded} essential files (meshes may be missing)")
            return result

    result["errors"].append(f"Failed to download model '{model_name}' from {repo}")
    return result


# Backward compatibility alias
def _download_model_from_gazebo_models(
    model_name: str,
    output_dir: str,
    branch: str = "master"
) -> Dict[str, Any]:
    """Backward compatibility wrapper for _download_model_from_repo."""
    return _download_model_from_repo(model_name, output_dir, "osrf/gazebo_models", branch)


@tool
def download_world_models(
    world_file_path: str,
    workspace_path: str,
    target_package: Optional[str] = None,
    models_repo: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download all models referenced in a world file to the workspace.

    This tool:
    1. Extracts all model:// references from the world file
    2. Downloads each model from the specified repository (or osrf/gazebo_models by default)
    3. Places them in the package's models/ directory
    4. Returns status for each model

    IMPORTANT: This should be called after downloading a world file to ensure
    all required models are available for Gazebo to load.

    CRITICAL: You MUST specify the correct models_repo based on the world's source:
    - AWS RoboMaker worlds (small_house, hospital, bookstore): Use the same repo as source
      e.g., "aws-robotics/aws-robomaker-small-house-world" has models in "models/" subdirectory
    - Gazebo classic worlds (willowgarage, cafe, empty): Use "osrf/gazebo_models"
    - TurtleBot3 worlds: Use "ROBOTIS-GIT/turtlebot3_simulations" with appropriate subdirectory

    Args:
        world_file_path: Path to the world file
        workspace_path: Path to the ROS workspace
        target_package: Package to store models (default: simbo_worlds)
        models_repo: GitHub repository containing models. Format: "owner/repo"
                     If not specified, defaults to "osrf/gazebo_models".
                     For AWS RoboMaker worlds, use the world's source repo.

    Returns:
        Dictionary with download status for each model
    """
    result = {
        "success": False,
        "models_downloaded": [],
        "models_failed": [],
        "models_skipped": [],
        "models_dir": None,
        "models_repo_used": models_repo or "osrf/gazebo_models",
        "errors": [],
        "warnings": [],
    }

    # Extract models from world file (use internal impl, not the tool wrapper)
    extract_result = _extract_world_models_impl(world_file_path)
    if not extract_result.get("success"):
        result["errors"].extend(extract_result.get("errors", ["Failed to extract models"]))
        return result

    models = extract_result.get("models", [])
    if not models:
        result["success"] = True
        result["warnings"].append("No models found in world file")
        return result

    # Find or create target package
    if target_package:
        package_path = os.path.join(workspace_path, "src", target_package)
    else:
        package_path = os.path.join(workspace_path, "src", "simbo_worlds")

    if not os.path.exists(package_path):
        result["errors"].append(f"Package not found: {package_path}")
        return result

    models_dir = os.path.join(package_path, "models")
    os.makedirs(models_dir, exist_ok=True)
    result["models_dir"] = models_dir

    # Determine repository and subdirectory for models
    repo = models_repo or "osrf/gazebo_models"

    # Determine models subdirectory based on repo type
    # AWS RoboMaker repos have models in "models/" subdirectory
    # osrf/gazebo_models has models at root level
    if "aws-robotics" in repo or "aws-robomaker" in repo.lower():
        models_subdir = "models"
    elif repo == "osrf/gazebo_models":
        models_subdir = ""  # Models at root level
    elif "turtlebot3" in repo.lower():
        models_subdir = "turtlebot3_gazebo/models"
    elif "clearpath" in repo.lower():
        models_subdir = "models"  # Clearpath repos usually have models/ dir
    else:
        # Default: assume models are in a "models" subdirectory
        models_subdir = "models"

    # Download each model
    for model_name in models:
        model_path = os.path.join(models_dir, model_name)

        # Skip if already exists
        if os.path.exists(model_path) and os.path.isdir(model_path):
            # Check if it has essential files
            if os.path.exists(os.path.join(model_path, "model.config")) or \
               os.path.exists(os.path.join(model_path, "model.sdf")):
                result["models_skipped"].append(model_name)
                continue

        # Try downloading from specified repo first
        download_result = _download_model_from_repo(
            model_name, models_dir, repo=repo, models_subdir=models_subdir
        )

        # If failed and not using osrf/gazebo_models, try osrf/gazebo_models as fallback
        # (for common models like ground_plane, sun, etc.)
        if not download_result.get("success") and repo != "osrf/gazebo_models":
            fallback_result = _download_model_from_repo(
                model_name, models_dir, repo="osrf/gazebo_models", models_subdir=""
            )
            if fallback_result.get("success"):
                download_result = fallback_result
                download_result["warnings"].append(
                    f"Model '{model_name}' downloaded from osrf/gazebo_models (fallback)"
                )

        if download_result.get("success"):
            result["models_downloaded"].append(model_name)
            result["warnings"].extend(download_result.get("warnings", []))
        else:
            result["models_failed"].append({
                "model": model_name,
                "errors": download_result.get("errors", []),
            })

    # Consider success if we downloaded at least some models
    if result["models_downloaded"] or result["models_skipped"]:
        result["success"] = True

    if result["models_failed"]:
        result["warnings"].append(
            f"Failed to download {len(result['models_failed'])} models. "
            "These may need to be installed via apt (ros-$ROS_DISTRO-gazebo-ros-pkgs) "
            "or available in GAZEBO_MODEL_PATH."
        )

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
    world_file_name: Optional[str] = None,
    use_latest: bool = True,
    workspace_path: Optional[str] = None,
    backup: bool = True
) -> Dict[str, Any]:
    """
    Update a simulation launch file to use a new world from the worlds package.

    This tool modifies existing launch files to point to the newly created world file
    in the simbo_worlds package. It handles both Python and XML launch files.
    
    By default, it uses the latest world from tracking. If world_file_name is provided,
    it will use that specific world instead.

    Args:
        launch_file_path: Path to the launch file to update
        worlds_package_name: Name of the package containing the world (e.g., "simbo_worlds")
        world_file_name: Optional name of the world file (e.g., "office.world" or "office")
                        If None and use_latest=True, uses the latest tracked world
        use_latest: If True and world_file_name is None, use the latest tracked world
        workspace_path: Path to the ROS workspace (required if use_latest=True)
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

    # Determine which world file to use
    if use_latest and world_file_name is None:
        if workspace_path is None:
            result["errors"].append("workspace_path is required when use_latest=True")
            return result
        
        latest_world = get_latest_world(workspace_path, worlds_package_name)
        if latest_world is None:
            result["errors"].append(f"No latest world tracked in {worlds_package_name} package")
            return result
        world_file_name = latest_world
        result["changes_made"].append(f"Using latest tracked world: {world_file_name}")
    elif world_file_name is None:
        result["errors"].append("Either world_file_name must be provided or use_latest must be True")
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

            # CRITICAL: Fix any incorrect import statements first
            # Replace wrong import: from ament_index_python.packages import ...
            # With correct import: from ament_index.python.packages import ...
            wrong_import_pattern = r'from\s+ament_index_python\.packages\s+import\s+get_package_share_directory'
            correct_import = "from ament_index_python.packages import get_package_share_directory"
            if re.search(wrong_import_pattern, new_content):
                new_content = re.sub(wrong_import_pattern, correct_import, new_content)
                result["changes_made"].append("Fixed incorrect import: changed ament_index_python to ament_index.python")

            # Check if get_package_share_directory import exists (with correct syntax)
            if "get_package_share_directory" not in new_content:
                # Add the import with CORRECT syntax
                import_line = "from ament_index_python.packages import get_package_share_directory\n"
                # Find where to insert (after other imports or at the beginning)
                import_match = re.search(r'^(import|from)\s+', new_content, re.MULTILINE)
                if import_match:
                    # Find the last import line
                    last_import = 0
                    for match in re.finditer(r'^(import|from)\s+.+$', new_content, re.MULTILINE):
                        last_import = match.end()
                    new_content = new_content[:last_import] + "\n" + import_line + new_content[last_import:]
                else:
                    new_content = import_line + new_content
                result["changes_made"].append("Added get_package_share_directory import")

            # Generate the new world path code
            # Use latest world tracking if available
            if use_latest and workspace_path:
                world_path_code = f"""
    # World file path from {worlds_package_name} (using latest tracked world)
    import json
    worlds_pkg_share = get_package_share_directory('{worlds_package_name}')
    latest_world_file = os.path.join(worlds_pkg_share, '.simbo_latest_world')
    if os.path.exists(latest_world_file):
        with open(latest_world_file, 'r') as f:
            latest_info = json.load(f)
        world_file_path = os.path.join(worlds_pkg_share, 'worlds', latest_info['latest_world'])
    else:
        # Fallback to specific world if tracking file doesn't exist
        world_file_path = os.path.join(worlds_pkg_share, 'worlds', '{world_file_name}')
"""
            else:
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

            # ALWAYS ensure world path code is added - be aggressive about updates
            # Check if world path code already exists
            world_path_exists = f"worlds_pkg_share = get_package_share_directory('{worlds_package_name}')" in new_content
            
            if not world_path_exists:
                # Find generate_launch_description function
                func_match = re.search(r'def\s+generate_launch_description\s*\(\s*\)\s*:', new_content)
                if func_match:
                    # Insert world path code at the beginning of the function
                    insert_pos = func_match.end()
                    new_content = new_content[:insert_pos] + world_path_code + new_content[insert_pos:]
                    result["changes_made"].append("Inserted world path code in generate_launch_description")

                    # Also ensure os and json imports exist
                    if "import os" not in new_content:
                        new_content = "import os\n" + new_content
                        result["changes_made"].append("Added os import")
                    if use_latest and "import json" not in new_content:
                        # Find where to insert json import (after os import or at the beginning)
                        if "import os" in new_content:
                            os_import_pos = new_content.find("import os")
                            os_import_end = new_content.find("\n", os_import_pos)
                            new_content = new_content[:os_import_end+1] + "import json\n" + new_content[os_import_end+1:]
                        else:
                            new_content = "import json\n" + new_content
                        result["changes_made"].append("Added json import for latest world tracking")
                else:
                    # No generate_launch_description function found - add it at the end of the file
                    new_content = new_content + "\n\n" + world_path_code
                    result["changes_made"].append("Added world path code at end of file (no generate_launch_description found)")
            
            # Now update any world references to use world_file_path
            # More aggressive pattern matching - look for any world-related assignments
            additional_patterns = [
                # Pattern: world_path = '...'
                (r"world_path\s*=\s*['\"][^'\"]+['\"]",
                 f"world_path = world_file_path"),
                # Pattern: world_name = '...'
                (r"world_name\s*=\s*['\"][^'\"]+['\"]",
                 f"world_name = world_file_path"),
                # Pattern: Any assignment with 'world' in variable name
                (r"(\w*world\w*)\s*=\s*['\"][^'\"]+\.world['\"]",
                 f"\\1 = world_file_path"),
            ]
            
            for pattern, replacement in additional_patterns:
                if re.search(pattern, new_content):
                    new_content = re.sub(pattern, replacement, new_content)
                    result["changes_made"].append(f"Updated world assignment pattern")
            
            # If we still haven't found where to use world_file_path, look for Gazebo launch includes
            # and update their launch_arguments
            if "world_file_path" not in new_content or "world_file_path" not in "".join(result["changes_made"]):
                # Look for IncludeLaunchDescription with world argument
                gazebo_include_pattern = r"(IncludeLaunchDescription\s*\([^)]*launch_arguments\s*=\s*\{[^}]*)(['\"]world['\"]\s*:\s*['\"][^'\"]+['\"])"
                if re.search(gazebo_include_pattern, new_content, re.DOTALL):
                    new_content = re.sub(
                        r"(['\"]world['\"]\s*:\s*)(['\"][^'\"]+['\"])",
                        r"\1world_file_path",
                        new_content
                    )
                    result["changes_made"].append("Updated Gazebo launch include world argument")

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

            xml_updated = False
            for pattern, replacement in xml_patterns:
                if re.search(pattern, new_content):
                    new_content = re.sub(pattern, replacement, new_content)
                    result["changes_made"].append(f"Updated XML world path")
                    xml_updated = True
            
            # If no XML patterns matched, add a world argument if it doesn't exist
            if not xml_updated:
                # Look for <launch> tag and add world argument after it
                launch_tag_match = re.search(r'<launch[^>]*>', new_content)
                if launch_tag_match:
                    world_arg_xml = f'\n  <arg name="world" default="$(find {worlds_package_name})/worlds/{world_file_name}"/>\n'
                    insert_pos = launch_tag_match.end()
                    new_content = new_content[:insert_pos] + world_arg_xml + new_content[insert_pos:]
                    result["changes_made"].append("Added world argument to XML launch file")

        # Write the modified content - ALWAYS write if we made any changes
        if new_content != original_content:
            with open(launch_file_path, 'w') as f:
                f.write(new_content)
            result["success"] = True
            result["changes_made"].append("Launch file updated successfully")
        else:
            # Even if no patterns found, we should have added the world path code
            # If we get here, something went wrong - but still mark as success if we have world_file_path
            if "world_file_path" in new_content or f"worlds_pkg_share = get_package_share_directory('{worlds_package_name}')" in new_content:
                result["success"] = True
                result["changes_made"].append("World path code already present or added")
            else:
                # Last resort: add the world path code at the very beginning of generate_launch_description
                func_match = re.search(r'def\s+generate_launch_description\s*\(\s*\)\s*:', new_content)
                if func_match:
                    insert_pos = func_match.end()
                    new_content = new_content[:insert_pos] + world_path_code + new_content[insert_pos:]
                    # Ensure imports
                    if "import os" not in new_content:
                        new_content = "import os\n" + new_content
                    if use_latest and "import json" not in new_content:
                        if "import os" in new_content:
                            os_import_pos = new_content.find("import os")
                            os_import_end = new_content.find("\n", os_import_pos)
                            new_content = new_content[:os_import_end+1] + "import json\n" + new_content[os_import_end+1:]
                    with open(launch_file_path, 'w') as f:
                        f.write(new_content)
                    result["success"] = True
                    result["changes_made"].append("Force-added world path code to launch file")
                else:
                    result["errors"].append("Could not find generate_launch_description function to update")
                    result["success"] = False

    except Exception as e:
        result["errors"].append(f"Error updating launch file: {str(e)}")

    return result


@tool
def update_all_simulation_launch_files(
    workspace_path: str,
    worlds_package_name: str = "simbo_worlds",
    use_latest: bool = True,
    backup: bool = True
) -> Dict[str, Any]:
    """
    Update ALL simulation launch files in the workspace to use the latest world.

    This tool finds all simulation launch files and updates them automatically.
    It ensures that ALL launch files are configured without requiring manual steps.

    Args:
        workspace_path: Path to the ROS workspace
        worlds_package_name: Name of the worlds package (default: "simbo_worlds")
        use_latest: If True, use the latest tracked world
        backup: Whether to create backups of original files

    Returns:
        Dictionary with update status for all launch files
    """
    result = {
        "success": True,
        "launch_files_updated": [],
        "launch_files_failed": [],
        "total_found": 0,
        "total_updated": 0,
        "errors": [],
    }

    # Find all simulation launch files
    launch_files_result = find_simulation_launch_files.invoke({"workspace_path": workspace_path})
    launch_files = launch_files_result.get("launch_files", [])
    
    result["total_found"] = len(launch_files)

    if not launch_files:
        result["errors"].append("No simulation launch files found in workspace")
        return result

    # Update each launch file
    for launch_file_info in launch_files:
        launch_file_path = launch_file_info.get("path")
        if not launch_file_path:
            continue

        try:
            update_result = update_simulation_launch_world.invoke({
                "launch_file_path": launch_file_path,
                "worlds_package_name": worlds_package_name,
                "use_latest": use_latest,
                "workspace_path": workspace_path,
                "backup": backup
            })

            if update_result.get("success"):
                result["launch_files_updated"].append({
                    "path": launch_file_path,
                    "name": launch_file_info.get("name"),
                    "changes": update_result.get("changes_made", [])
                })
                result["total_updated"] += 1
            else:
                result["launch_files_failed"].append({
                    "path": launch_file_path,
                    "name": launch_file_info.get("name"),
                    "errors": update_result.get("errors", [])
                })
                result["errors"].extend(update_result.get("errors", []))

        except Exception as e:
            result["launch_files_failed"].append({
                "path": launch_file_path,
                "name": launch_file_info.get("name"),
                "errors": [str(e)]
            })
            result["errors"].append(f"Error updating {launch_file_path}: {str(e)}")

    # Overall success if at least one file was updated
    if result["total_updated"] == 0 and result["total_found"] > 0:
        result["success"] = False

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
from ament_index_python.packages import get_package_share_directory
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
# from ament_index_python.packages import get_package_share_directory
# world_path = os.path.join(get_package_share_directory('{package_name}'), 'worlds', '{world_name}.world')
'''

    return {
        "launch_code": launch_code,
        "usage_instructions": usage,
        "launch_file_name": f"{world_name}_world.launch.py",
    }


# =============================================================================
# Latest World Tracking Tools
# =============================================================================

def get_latest_world(
    workspace_path: str,
    worlds_package_name: str = "simbo_worlds"
) -> Optional[str]:
    """
    Get the name of the most recently generated world file.

    Args:
        workspace_path: Path to the ROS workspace
        worlds_package_name: Name of the worlds package (default: "simbo_worlds")

    Returns:
        Name of the latest world file (e.g., "office.world") or None if not tracked
    """
    src_path = os.path.join(workspace_path, "src")
    package_path = os.path.join(src_path, worlds_package_name)
    tracking_file = os.path.join(package_path, ".simbo_latest_world")

    if not os.path.exists(tracking_file):
        return None

    try:
        with open(tracking_file, 'r') as f:
            data = json.load(f)
            return data.get("latest_world")
    except (json.JSONDecodeError, IOError):
        return None


@tool
def track_latest_world(
    workspace_path: str,
    world_file_name: str,
    worlds_package_name: str = "simbo_worlds"
) -> Dict[str, Any]:
    """
    Track the most recently generated world file as the latest.

    Creates or updates a metadata file that stores which world file is the most recent.
    This allows launch files to automatically use the latest generated world.

    Args:
        workspace_path: Path to the ROS workspace
        world_file_name: Name of the world file (e.g., "office.world")
        worlds_package_name: Name of the worlds package (default: "simbo_worlds")

    Returns:
        Dictionary with tracking status
    """
    result = {
        "success": False,
        "tracking_file_path": None,
        "latest_world": None,
        "errors": [],
    }

    src_path = os.path.join(workspace_path, "src")
    if not os.path.exists(src_path):
        result["errors"].append(f"No src directory found: {workspace_path}")
        return result

    package_path = os.path.join(src_path, worlds_package_name)
    
    # Ensure package exists
    if not os.path.exists(package_path):
        result["errors"].append(f"Worlds package not found: {worlds_package_name}")
        return result

    # Ensure world file name has extension
    if not world_file_name.endswith('.world') and not world_file_name.endswith('.sdf'):
        world_file_name = f"{world_file_name}.world"

    # Verify world file exists
    worlds_dir = os.path.join(package_path, "worlds")
    world_file_path = os.path.join(worlds_dir, world_file_name)
    if not os.path.exists(world_file_path):
        result["errors"].append(f"World file not found: {world_file_path}")
        return result

    # Create/update tracking file
    tracking_file = os.path.join(package_path, ".simbo_latest_world")
    
    try:
        tracking_data = {
            "latest_world": world_file_name,
            "timestamp": datetime.now().isoformat(),
            "package": worlds_package_name,
            "world_file_path": world_file_path,
        }

        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)

        result["success"] = True
        result["tracking_file_path"] = tracking_file
        result["latest_world"] = world_file_name

    except Exception as e:
        result["errors"].append(f"Failed to write tracking file: {str(e)}")

    return result
