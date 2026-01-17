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
def check_setup_cfg(package_path: str) -> Dict[str, any]:
    """
    Check if a ROS2 Python package has a proper setup.cfg file.

    The setup.cfg file is CRITICAL for ROS2 Python packages because it tells
    colcon where to install the executable scripts. Without it, executables
    are installed to bin/ instead of lib/<package_name>/, and ros2 run will
    fail with "No executable found".

    Args:
        package_path: Path to the ROS2 package directory

    Returns:
        Dictionary with validation results and the correct setup.cfg content
    """
    result = {
        "exists": False,
        "is_valid": False,
        "has_scripts_dir": False,
        "has_install_scripts": False,
        "current_content": None,
        "correct_content": None,
        "errors": [],
        "fix_suggestions": []
    }

    setup_cfg_path = os.path.join(package_path, "setup.cfg")
    package_name = os.path.basename(package_path)

    # The correct setup.cfg content for ROS2 Python packages
    correct_content = f"""[develop]
script_dir=$base/lib/{package_name}
[install]
install_scripts=$base/lib/{package_name}
"""
    result["correct_content"] = correct_content

    if not os.path.exists(setup_cfg_path):
        result["errors"].append(f"setup.cfg not found in {package_path}")
        result["fix_suggestions"].append(
            f"Create setup.cfg with content:\n{correct_content}"
        )
        return result

    result["exists"] = True

    try:
        with open(setup_cfg_path, 'r') as f:
            content = f.read()
        result["current_content"] = content

        # Check for the critical install_scripts directive
        if "install_scripts" in content:
            result["has_install_scripts"] = True
            # Verify it points to the correct location
            expected_path = f"lib/{package_name}"
            if expected_path in content:
                result["is_valid"] = True
            else:
                result["errors"].append(
                    f"install_scripts does not point to lib/{package_name}"
                )
                result["fix_suggestions"].append(
                    f"Update setup.cfg install_scripts to: $base/lib/{package_name}"
                )
        else:
            result["errors"].append("setup.cfg missing install_scripts directive")
            result["fix_suggestions"].append(
                f"Add to setup.cfg [install] section: install_scripts=$base/lib/{package_name}"
            )

        # Check for script_dir in develop section
        if "script_dir" in content:
            result["has_scripts_dir"] = True

    except Exception as e:
        result["errors"].append(f"Error reading setup.cfg: {str(e)}")

    return result


@tool
def create_setup_cfg(package_path: str) -> Dict[str, any]:
    """
    Create or fix the setup.cfg file for a ROS2 Python package.

    This is CRITICAL for ROS2 Python packages. Without setup.cfg, executables
    are installed to the wrong location and ros2 run will fail.

    Args:
        package_path: Path to the ROS2 package directory

    Returns:
        Dictionary with creation/update status
    """
    result = {
        "success": False,
        "created": False,
        "updated": False,
        "path": None,
        "content": None,
        "error": None
    }

    setup_cfg_path = os.path.join(package_path, "setup.cfg")
    package_name = os.path.basename(package_path)

    # The correct setup.cfg content
    correct_content = f"""[develop]
script_dir=$base/lib/{package_name}
[install]
install_scripts=$base/lib/{package_name}
"""

    result["path"] = setup_cfg_path
    result["content"] = correct_content

    try:
        file_existed = os.path.exists(setup_cfg_path)

        with open(setup_cfg_path, 'w') as f:
            f.write(correct_content)

        result["success"] = True
        if file_existed:
            result["updated"] = True
        else:
            result["created"] = True

    except Exception as e:
        result["error"] = f"Error writing setup.cfg: {str(e)}"

    return result


@tool
def create_ros2_python_package(
    workspace_path: str,
    package_name: str,
    node_name: str,
    description: str = "A ROS2 Python package",
    maintainer: str = "user",
    maintainer_email: str = "user@example.com",
    dependencies: List[str] = None
) -> Dict[str, any]:
    """
    Create a complete ROS2 Python package structure with all required files.

    This creates the CORRECT ament_python package structure:
    - package.xml (with ament_python build type)
    - setup.py (with entry_points)
    - setup.cfg (CRITICAL for executable installation)
    - resource/<package_name> (ament marker)
    - <package_name>/__init__.py
    - <package_name>/<node_name>.py (template node)

    NO CMakeLists.txt is created - this is a Python-only package!

    Args:
        workspace_path: Path to the ROS workspace
        package_name: Name of the package to create
        node_name: Name of the main node (without .py extension)
        description: Package description
        maintainer: Package maintainer name
        maintainer_email: Maintainer email
        dependencies: List of ROS2 dependencies (default: ['rclpy', 'std_msgs'])

    Returns:
        Dictionary with creation status and file paths
    """
    if dependencies is None:
        dependencies = ['rclpy', 'std_msgs']

    result = {
        "success": False,
        "package_path": None,
        "files_created": [],
        "errors": []
    }

    # Package path
    src_path = os.path.join(workspace_path, "src")
    package_path = os.path.join(src_path, package_name)
    result["package_path"] = package_path

    try:
        # Create directory structure
        os.makedirs(package_path, exist_ok=True)
        os.makedirs(os.path.join(package_path, package_name), exist_ok=True)
        os.makedirs(os.path.join(package_path, "resource"), exist_ok=True)

        # 1. Create package.xml (ament_python - NO CMake!)
        deps_xml = "\n  ".join([f"<depend>{dep}</depend>" for dep in dependencies])
        package_xml_content = f"""<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{package_name}</name>
  <version>0.0.1</version>
  <description>{description}</description>
  <maintainer email="{maintainer_email}">{maintainer}</maintainer>
  <license>Apache-2.0</license>

  {deps_xml}

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
"""
        package_xml_path = os.path.join(package_path, "package.xml")
        with open(package_xml_path, 'w') as f:
            f.write(package_xml_content)
        result["files_created"].append(package_xml_path)

        # 2. Create setup.py with entry_points
        setup_py_content = f"""from setuptools import find_packages, setup

package_name = '{package_name}'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='{maintainer}',
    maintainer_email='{maintainer_email}',
    description='{description}',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={{
        'console_scripts': [
            '{node_name}={package_name}.{node_name}:main',
        ],
    }},
)
"""
        setup_py_path = os.path.join(package_path, "setup.py")
        with open(setup_py_path, 'w') as f:
            f.write(setup_py_content)
        result["files_created"].append(setup_py_path)

        # 3. Create setup.cfg (CRITICAL!)
        setup_cfg_content = f"""[develop]
script_dir=$base/lib/{package_name}
[install]
install_scripts=$base/lib/{package_name}
"""
        setup_cfg_path = os.path.join(package_path, "setup.cfg")
        with open(setup_cfg_path, 'w') as f:
            f.write(setup_cfg_content)
        result["files_created"].append(setup_cfg_path)

        # 4. Create resource marker file
        resource_path = os.path.join(package_path, "resource", package_name)
        with open(resource_path, 'w') as f:
            f.write("")  # Empty file
        result["files_created"].append(resource_path)

        # 5. Create __init__.py
        init_path = os.path.join(package_path, package_name, "__init__.py")
        with open(init_path, 'w') as f:
            f.write("")  # Empty file
        result["files_created"].append(init_path)

        # 6. Create template node file
        node_content = f'''#!/usr/bin/env python3
"""
{node_name} - A ROS2 node.

This is a template node created by Simbo.
"""

import rclpy
from rclpy.node import Node


class {node_name.title().replace("_", "")}Node(Node):
    """A ROS2 node."""

    def __init__(self):
        super().__init__('{node_name}')
        self.get_logger().info('{node_name} node started')

        # TODO: Add your node logic here
        # Example timer:
        # self.timer = self.create_timer(1.0, self.timer_callback)

        # Example publisher:
        # self.publisher = self.create_publisher(String, 'topic_name', 10)

        # Example subscriber:
        # self.subscription = self.create_subscription(
        #     String, 'topic_name', self.listener_callback, 10)

    def timer_callback(self):
        """Timer callback - called periodically."""
        pass


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = {node_name.title().replace("_", "")}Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
'''
        node_path = os.path.join(package_path, package_name, f"{node_name}.py")
        with open(node_path, 'w') as f:
            f.write(node_content)
        result["files_created"].append(node_path)

        result["success"] = True

    except Exception as e:
        result["errors"].append(f"Error creating package: {str(e)}")

    return result


@tool
def check_executable_configuration(
    workspace_path: str,
    node_file: str,
    ros_version: str = "ros2"
) -> Dict[str, any]:
    """
    Check if a ROS node file has proper executable configuration.
    This is a high-level validation that checks entry points, setup.cfg (ROS2), and permissions (ROS1).

    For ROS2 Python packages, THREE things are required:
    1. Entry points in setup.py (console_scripts)
    2. setup.cfg with install_scripts pointing to lib/<package_name>/
    3. The node file must exist in the package

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
        # For ROS2, check entry points in setup.py AND setup.cfg
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

        entry_points_valid = False
        setup_cfg_valid = False

        # Check 1: Entry points in setup.py
        node_name = os.path.splitext(os.path.basename(node_file))[0]
        entry_point_result = check_ros2_entry_points.invoke({
            "package_path": package_dir,
            "node_name": node_name,
            "node_file": node_file
        })

        if entry_point_result.get("errors"):
            result["errors"].extend(entry_point_result["errors"])

        if entry_point_result.get("missing_entry_points"):
            result["errors"].append(
                f"Missing entry point for node '{node_name}' in setup.py"
            )
            if entry_point_result.get("suggestions"):
                result["fix_suggestions"].extend(entry_point_result["suggestions"])
        elif entry_point_result.get("has_entry_points"):
            entry_points_valid = True

        # Check 2: setup.cfg exists and is properly configured
        setup_cfg_result = check_setup_cfg.invoke({"package_path": package_dir})

        if not setup_cfg_result.get("exists"):
            result["errors"].append(
                f"CRITICAL: setup.cfg missing in {package_dir}. "
                "Without this file, ros2 run will fail with 'No executable found'."
            )
            result["fix_suggestions"].append(
                f"Create setup.cfg in {package_dir} with content:\n{setup_cfg_result.get('correct_content', '')}"
            )
        elif not setup_cfg_result.get("is_valid"):
            result["errors"].extend(setup_cfg_result.get("errors", []))
            result["fix_suggestions"].extend(setup_cfg_result.get("fix_suggestions", []))
        else:
            setup_cfg_valid = True

        # Both must be valid for ROS2
        result["is_valid"] = entry_points_valid and setup_cfg_valid

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
