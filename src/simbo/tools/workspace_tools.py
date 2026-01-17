"""
Tools for analyzing and interacting with ROS workspaces.
"""

import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from langchain_core.tools import tool


@tool
def detect_ros_version(workspace_path: str) -> Dict[str, str]:
    """
    Detect ROS version and distribution in the given workspace or system.

    Args:
        workspace_path: Path to the ROS workspace

    Returns:
        Dictionary with ros_version, ros_distro, and gazebo_version
    """
    result = {
        "ros_version": "unknown",
        "ros_distro": "unknown",
        "gazebo_version": "unknown",
        "workspace_type": "unknown"
    }

    # Check for ROS2 indicators
    ros2_indicators = [
        os.path.join(workspace_path, "install"),
        os.path.join(workspace_path, "build"),
        os.path.join(workspace_path, "log"),
    ]

    # Check for ROS1 indicators
    ros1_indicators = [
        os.path.join(workspace_path, "devel"),
        os.path.join(workspace_path, "build"),
    ]

    # Detect workspace type
    has_ros2_structure = all(os.path.exists(p) for p in ros2_indicators[:2])
    has_ros1_structure = all(os.path.exists(p) for p in ros1_indicators)

    if has_ros2_structure and os.path.exists(os.path.join(workspace_path, "install")):
        result["workspace_type"] = "ros2_colcon"
    elif has_ros1_structure and os.path.exists(os.path.join(workspace_path, "devel")):
        result["workspace_type"] = "ros1_catkin"
    elif os.path.exists(os.path.join(workspace_path, "src")):
        result["workspace_type"] = "ros_workspace"

    # Try to detect ROS version from environment
    ros_distro = os.environ.get("ROS_DISTRO", "")
    ros_version = os.environ.get("ROS_VERSION", "")

    if ros_distro:
        result["ros_distro"] = ros_distro

    if ros_version:
        result["ros_version"] = f"ros{ros_version}"
    elif ros_distro:
        # Infer from distro name
        ros2_distros = ["humble", "iron", "jazzy", "rolling", "foxy", "galactic"]
        ros1_distros = ["noetic", "melodic", "kinetic"]
        if ros_distro.lower() in ros2_distros:
            result["ros_version"] = "ros2"
        elif ros_distro.lower() in ros1_distros:
            result["ros_version"] = "ros1"

    # Try to detect Gazebo version
    try:
        gz_result = subprocess.run(
            ["gz", "--version"], capture_output=True, text=True, timeout=5
        )
        if gz_result.returncode == 0:
            result["gazebo_version"] = gz_result.stdout.strip().split("\n")[0]
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            gazebo_result = subprocess.run(
                ["gazebo", "--version"], capture_output=True, text=True, timeout=5
            )
            if gazebo_result.returncode == 0:
                result["gazebo_version"] = gazebo_result.stdout.strip().split("\n")[0]
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    return result


@tool
def analyze_workspace(workspace_path: str) -> Dict[str, any]:
    """
    Perform comprehensive analysis of a ROS workspace.

    Args:
        workspace_path: Path to the ROS workspace

    Returns:
        Dictionary containing workspace analysis results
    """
    analysis = {
        "path": workspace_path,
        "exists": os.path.exists(workspace_path),
        "packages": [],
        "source_files": {},
        "launch_files": [],
        "config_files": [],
        "urdf_files": [],
        "world_files": [],
        "structure": {},
        "errors": []
    }

    if not analysis["exists"]:
        analysis["errors"].append(f"Workspace path does not exist: {workspace_path}")
        return analysis

    src_path = os.path.join(workspace_path, "src")
    if not os.path.exists(src_path):
        analysis["errors"].append("No 'src' directory found in workspace")
        return analysis

    # Find all packages
    for root, dirs, files in os.walk(src_path):
        if "package.xml" in files:
            package_path = root
            package_name = os.path.basename(package_path)
            analysis["packages"].append(package_name)

            # Find source files in package
            pkg_sources = []
            for ext in [".py", ".cpp", ".c", ".h", ".hpp"]:
                for src_root, _, src_files in os.walk(package_path):
                    for f in src_files:
                        if f.endswith(ext):
                            pkg_sources.append(os.path.join(src_root, f))
            analysis["source_files"][package_name] = pkg_sources

    # Find launch files
    for root, dirs, files in os.walk(src_path):
        for f in files:
            file_path = os.path.join(root, f)
            if f.endswith(".launch") or f.endswith(".launch.py") or f.endswith(".launch.xml"):
                analysis["launch_files"].append(file_path)
            elif f.endswith(".yaml") or f.endswith(".yml"):
                analysis["config_files"].append(file_path)
            elif f.endswith(".urdf") or f.endswith(".xacro"):
                analysis["urdf_files"].append(file_path)
            elif f.endswith(".world") or f.endswith(".sdf"):
                analysis["world_files"].append(file_path)

    return analysis


@tool
def list_packages(workspace_path: str) -> List[Dict[str, str]]:
    """
    List all ROS packages in the workspace with their descriptions.

    Args:
        workspace_path: Path to the ROS workspace

    Returns:
        List of dictionaries with package information
    """
    packages = []
    src_path = os.path.join(workspace_path, "src")

    if not os.path.exists(src_path):
        return packages

    for root, dirs, files in os.walk(src_path):
        if "package.xml" in files:
            package_info = {
                "name": os.path.basename(root),
                "path": root,
                "description": "",
                "dependencies": [],
                "type": "unknown"
            }

            # Parse package.xml
            try:
                tree = ET.parse(os.path.join(root, "package.xml"))
                pkg_root = tree.getroot()

                desc_elem = pkg_root.find("description")
                if desc_elem is not None and desc_elem.text:
                    package_info["description"] = desc_elem.text.strip()

                # Get dependencies
                for dep_tag in ["depend", "exec_depend", "build_depend"]:
                    for dep in pkg_root.findall(dep_tag):
                        if dep.text and dep.text not in package_info["dependencies"]:
                            package_info["dependencies"].append(dep.text)

                # Determine package type
                if os.path.exists(os.path.join(root, "setup.py")):
                    package_info["type"] = "ament_python"
                elif os.path.exists(os.path.join(root, "CMakeLists.txt")):
                    package_info["type"] = "ament_cmake"

            except ET.ParseError:
                package_info["description"] = "Error parsing package.xml"

            packages.append(package_info)

    return packages


@tool
def read_package_xml(package_path: str) -> Dict[str, any]:
    """
    Read and parse a package.xml file.

    Args:
        package_path: Path to the package directory

    Returns:
        Dictionary with package metadata
    """
    xml_path = os.path.join(package_path, "package.xml")
    result = {
        "exists": False,
        "name": "",
        "version": "",
        "description": "",
        "maintainers": [],
        "dependencies": [],
        "build_type": "",
        "raw_xml": ""
    }

    if not os.path.exists(xml_path):
        return result

    result["exists"] = True

    try:
        with open(xml_path, "r") as f:
            result["raw_xml"] = f.read()

        tree = ET.parse(xml_path)
        root = tree.getroot()

        name_elem = root.find("name")
        if name_elem is not None:
            result["name"] = name_elem.text

        version_elem = root.find("version")
        if version_elem is not None:
            result["version"] = version_elem.text

        desc_elem = root.find("description")
        if desc_elem is not None:
            result["description"] = desc_elem.text

        for maintainer in root.findall("maintainer"):
            result["maintainers"].append(maintainer.text)

        for dep_tag in ["depend", "exec_depend", "build_depend", "build_export_depend"]:
            for dep in root.findall(dep_tag):
                if dep.text and dep.text not in result["dependencies"]:
                    result["dependencies"].append(dep.text)

        build_type = root.find("build_type")
        if build_type is not None:
            result["build_type"] = build_type.text

    except ET.ParseError as e:
        result["error"] = str(e)

    return result


@tool
def find_launch_files(workspace_path: str, package_name: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Find all launch files in the workspace or specific package.

    Args:
        workspace_path: Path to the ROS workspace
        package_name: Optional specific package to search in

    Returns:
        List of dictionaries with launch file information
    """
    launch_files = []
    search_path = os.path.join(workspace_path, "src")

    if package_name:
        # Find specific package
        for root, dirs, files in os.walk(search_path):
            if os.path.basename(root) == package_name and "package.xml" in files:
                search_path = root
                break

    for root, dirs, files in os.walk(search_path):
        for f in files:
            if f.endswith(".launch") or f.endswith(".launch.py") or f.endswith(".launch.xml"):
                file_path = os.path.join(root, f)
                launch_info = {
                    "name": f,
                    "path": file_path,
                    "type": "python" if f.endswith(".py") else "xml",
                    "package": ""
                }

                # Determine which package this belongs to
                rel_path = os.path.relpath(file_path, search_path)
                parts = rel_path.split(os.sep)
                if len(parts) > 1:
                    launch_info["package"] = parts[0]

                launch_files.append(launch_info)

    return launch_files


@tool
def find_source_files(
    workspace_path: str,
    package_name: Optional[str] = None,
    file_type: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Find source files in the workspace or specific package.

    Args:
        workspace_path: Path to the ROS workspace
        package_name: Optional specific package to search in
        file_type: Optional file extension filter (e.g., 'py', 'cpp')

    Returns:
        List of dictionaries with source file information
    """
    source_files = []
    search_path = os.path.join(workspace_path, "src")

    # Define source extensions
    extensions = {
        "py": [".py"],
        "cpp": [".cpp", ".cc", ".cxx"],
        "c": [".c"],
        "header": [".h", ".hpp", ".hxx"],
        "all": [".py", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx"]
    }

    target_extensions = extensions.get(file_type, extensions["all"]) if file_type else extensions["all"]

    if package_name:
        for root, dirs, files in os.walk(search_path):
            if os.path.basename(root) == package_name and "package.xml" in files:
                search_path = root
                break

    for root, dirs, files in os.walk(search_path):
        # Skip build and install directories
        dirs[:] = [d for d in dirs if d not in ["build", "install", "log", "__pycache__", ".git"]]

        for f in files:
            if any(f.endswith(ext) for ext in target_extensions):
                file_path = os.path.join(root, f)
                file_info = {
                    "name": f,
                    "path": file_path,
                    "extension": os.path.splitext(f)[1],
                    "package": "",
                    "size": os.path.getsize(file_path)
                }

                # Determine package
                rel_path = os.path.relpath(file_path, os.path.join(workspace_path, "src"))
                parts = rel_path.split(os.sep)
                if len(parts) > 1:
                    file_info["package"] = parts[0]

                source_files.append(file_info)

    return source_files


@tool
def check_ros2_entry_points(
    package_path: str,
    node_name: Optional[str] = None,
    node_file: Optional[str] = None
) -> Dict[str, any]:
    """
    Check if a ROS2 package has proper entry points configured in setup.py.
    
    Args:
        package_path: Path to the ROS2 package directory
        node_name: Optional specific node name to check for
        node_file: Optional path to node file to extract node name from
    
    Returns:
        Dictionary with validation results and suggestions
    """
    result = {
        "setup_py_exists": False,
        "has_entry_points": False,
        "entry_points_found": [],
        "missing_entry_points": [],
        "errors": [],
        "suggestions": []
    }
    
    setup_py_path = os.path.join(package_path, "setup.py")
    
    if not os.path.exists(setup_py_path):
        result["errors"].append(f"setup.py not found in {package_path}")
        return result
    
    result["setup_py_exists"] = True
    
    # Extract node name from file if provided
    if node_file and not node_name:
        node_name = os.path.splitext(os.path.basename(node_file))[0]
    
    try:
        with open(setup_py_path, 'r') as f:
            content = f.read()
        
        # Check if entry_points section exists
        if "entry_points" in content and "console_scripts" in content:
            result["has_entry_points"] = True
            
            # Extract existing entry points
            # Pattern: 'node_name=' or "node_name="
            entry_point_pattern = re.compile(r"['\"]([^'\"]+)=([^'\"]+):([^'\"]+)['\"]")
            matches = entry_point_pattern.findall(content)
            
            for match in matches:
                ep_node_name, module_path, func_name = match
                result["entry_points_found"].append({
                    "node_name": ep_node_name,
                    "module_path": module_path,
                    "function": func_name
                })
            
            # If specific node name provided, check if it exists
            if node_name:
                found = any(ep["node_name"] == node_name for ep in result["entry_points_found"])
                if not found:
                    result["missing_entry_points"].append(node_name)
                    # Generate suggestion
                    if node_file:
                        # Try to extract module path
                        workspace_path = package_path
                        while not os.path.exists(os.path.join(workspace_path, "src")) and workspace_path != "/":
                            workspace_path = os.path.dirname(workspace_path)
                        
                        src_path = os.path.join(workspace_path, "src")
                        if os.path.exists(src_path):
                            rel_path = os.path.relpath(node_file, src_path)
                            module_path = rel_path.replace(os.sep, '.').replace('.py', '')
                        else:
                            # Fallback: use package name
                            package_name = os.path.basename(package_path)
                            node_base = os.path.splitext(os.path.basename(node_file))[0]
                            module_path = f"{package_name}.{node_base}"
                        
                        # Try to extract function name
                        func_name = "main"  # Default
                        try:
                            with open(node_file, 'r') as f:
                                node_content = f.read()
                                main_match = re.search(r'def\s+(main)\s*\(', node_content)
                                if main_match:
                                    func_name = main_match.group(1)
                        except Exception:
                            pass
                        
                        entry_point_str = f"{node_name}={module_path}:{func_name}"
                        result["suggestions"].append(
                            f"Add to setup.py entry_points: {{'console_scripts': ['{entry_point_str}']}}"
                        )
        else:
            result["errors"].append("setup.py does not contain entry_points with console_scripts")
            if node_name:
                result["suggestions"].append(
                    "Add entry_points section to setup.py with console_scripts"
                )
    
    except Exception as e:
        result["errors"].append(f"Error reading setup.py: {str(e)}")
    
    return result


@tool
def check_python_script_executable(script_path: str) -> Dict[str, any]:
    """
    Check if a Python script is executable.
    
    Args:
        script_path: Path to the Python script file
    
    Returns:
        Dictionary with executable status and fix suggestion
    """
    result = {
        "exists": False,
        "is_executable": False,
        "fix_command": None,
        "error": None
    }
    
    if not os.path.exists(script_path):
        result["error"] = f"Script file not found: {script_path}"
        return result
    
    result["exists"] = True
    
    try:
        stat_info = os.stat(script_path)
        is_executable = bool(stat_info.st_mode & 0o111)
        result["is_executable"] = is_executable
        
        if not is_executable:
            result["fix_command"] = f"chmod +x {script_path}"
    
    except Exception as e:
        result["error"] = f"Error checking file permissions: {str(e)}"
    
    return result


@tool
def check_executable_configuration(
    workspace_path: str,
    node_file: str,
    ros_version: str = "ros2"
) -> Dict[str, any]:
    """
    Check if a ROS node file has proper executable configuration.
    This is a high-level validation that checks both entry points (ROS2) and permissions (ROS1).
    
    Args:
        workspace_path: Path to the ROS workspace
        node_file: Path to the Python node file
        ros_version: Either "ros1" or "ros2"
    
    Returns:
        Dictionary with validation results, errors, and fix suggestions
    """
    result = {
        "is_valid": False,
        "node_file": node_file,
        "ros_version": ros_version,
        "errors": [],
        "warnings": [],
        "fix_suggestions": []
    }
    
    if not os.path.exists(node_file):
        result["errors"].append(f"Node file not found: {node_file}")
        return result
    
    if ros_version == "ros2":
        # For ROS2, check entry points in setup.py
        # Find package directory containing this node
        current_dir = os.path.dirname(node_file)
        package_dir = None
        
        # Walk up to find package.xml or setup.py
        while current_dir != workspace_path and current_dir != "/":
            if os.path.exists(os.path.join(current_dir, "package.xml")):
                package_dir = current_dir
                break
            current_dir = os.path.dirname(current_dir)
        
        if not package_dir:
            result["errors"].append(f"Could not find package directory for {node_file}")
            result["warnings"].append("Node might not be in a proper ROS package")
            return result
        
        # Check entry points
        node_name = os.path.splitext(os.path.basename(node_file))[0]
        entry_point_result = check_ros2_entry_points.invoke({
            "package_path": package_dir,
            "node_name": node_name,
            "node_file": node_file
        })
        
        if entry_point_result.get("errors"):
            result["errors"].extend(entry_point_result["errors"])
        
        if entry_point_result.get("missing_entry_points"):
            result["is_valid"] = False
            result["errors"].append(
                f"Missing entry point for node '{node_name}' in setup.py"
            )
            if entry_point_result.get("suggestions"):
                result["fix_suggestions"].extend(entry_point_result["suggestions"])
        elif entry_point_result.get("has_entry_points"):
            result["is_valid"] = True
    
    elif ros_version == "ros1":
        # For ROS1, check if script is executable
        exec_result = check_python_script_executable.invoke({"script_path": node_file})
        
        if exec_result.get("error"):
            result["errors"].append(exec_result["error"])
        
        if not exec_result.get("is_executable"):
            result["is_valid"] = False
            result["errors"].append(f"Python script is not executable: {node_file}")
            if exec_result.get("fix_command"):
                result["fix_suggestions"].append(exec_result["fix_command"])
        else:
            result["is_valid"] = True
    
    else:
        result["errors"].append(f"Unknown ROS version: {ros_version}")
    
    return result
