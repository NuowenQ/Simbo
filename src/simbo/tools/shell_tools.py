"""
Shell execution tools for running commands in the user's environment.
Allows the agent to build, test, and run ROS commands.
"""

import os
import subprocess
import shlex
from typing import Dict, Any, Optional, List
from langchain_core.tools import tool


@tool
def run_command(
    command: str,
    working_directory: Optional[str] = None,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Execute a shell command and return the output.
    Use this to run build commands, tests, or ROS commands.

    Args:
        command: The command to execute
        working_directory: Directory to run the command in (optional)
        timeout: Maximum time to wait for command completion in seconds

    Returns:
        Dictionary with command output, return code, and any errors
    """
    result = {
        "success": False,
        "command": command,
        "stdout": "",
        "stderr": "",
        "return_code": -1,
        "error": None
    }

    # Safety check - block dangerous commands
    dangerous_patterns = [
        "rm -rf /",
        "rm -rf ~",
        ":(){ :|:& };:",  # Fork bomb
        "> /dev/sda",
        "mkfs",
        "dd if=",
    ]

    for pattern in dangerous_patterns:
        if pattern in command:
            result["error"] = f"Blocked potentially dangerous command pattern: {pattern}"
            return result

    try:
        # Set working directory
        cwd = working_directory if working_directory and os.path.exists(working_directory) else None

        # Run the command
        process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

        result["stdout"] = process.stdout
        result["stderr"] = process.stderr
        result["return_code"] = process.returncode
        result["success"] = process.returncode == 0

    except subprocess.TimeoutExpired:
        result["error"] = f"Command timed out after {timeout} seconds"
    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def run_ros_command(
    command: str,
    ros_version: str = "ros2",
    workspace_path: Optional[str] = None,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Execute a ROS-specific command with proper environment setup.

    Args:
        command: The ROS command to execute (e.g., "ros2 topic list")
        ros_version: Either "ros1" or "ros2"
        workspace_path: Path to the ROS workspace (will source setup files)
        timeout: Maximum time to wait

    Returns:
        Dictionary with command output
    """
    result = {
        "success": False,
        "command": command,
        "stdout": "",
        "stderr": "",
        "return_code": -1,
        "error": None
    }

    try:
        # Build the full command with sourcing
        full_command_parts = []

        # Source ROS setup
        if ros_version == "ros2":
            ros_distro = os.environ.get("ROS_DISTRO", "humble")
            ros_setup = f"/opt/ros/{ros_distro}/setup.bash"
            if os.path.exists(ros_setup):
                full_command_parts.append(f"source {ros_setup}")
        else:
            ros_distro = os.environ.get("ROS_DISTRO", "noetic")
            ros_setup = f"/opt/ros/{ros_distro}/setup.bash"
            if os.path.exists(ros_setup):
                full_command_parts.append(f"source {ros_setup}")

        # Source workspace setup
        if workspace_path:
            if ros_version == "ros2":
                ws_setup = os.path.join(workspace_path, "install", "setup.bash")
            else:
                ws_setup = os.path.join(workspace_path, "devel", "setup.bash")

            if os.path.exists(ws_setup):
                full_command_parts.append(f"source {ws_setup}")

        full_command_parts.append(command)
        full_command = " && ".join(full_command_parts)

        process = subprocess.run(
            ["bash", "-c", full_command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workspace_path
        )

        result["stdout"] = process.stdout
        result["stderr"] = process.stderr
        result["return_code"] = process.returncode
        result["success"] = process.returncode == 0

    except subprocess.TimeoutExpired:
        result["error"] = f"Command timed out after {timeout} seconds"
    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def build_ros_workspace(
    workspace_path: str,
    ros_version: str = "ros2",
    packages: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Build a ROS workspace using colcon (ROS2) or catkin (ROS1).

    Args:
        workspace_path: Path to the ROS workspace
        ros_version: Either "ros1" or "ros2"
        packages: Optional list of specific packages to build

    Returns:
        Dictionary with build output and status
    """
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "error": None
    }

    if not os.path.exists(workspace_path):
        result["error"] = f"Workspace not found: {workspace_path}"
        return result

    try:
        if ros_version == "ros2":
            # Use colcon for ROS2
            cmd = "colcon build"
            if packages:
                cmd += f" --packages-select {' '.join(packages)}"
        else:
            # Use catkin_make for ROS1
            cmd = "catkin_make"
            if packages:
                cmd += f" --pkg {' '.join(packages)}"

        # Build full command with sourcing
        ros_distro = os.environ.get("ROS_DISTRO", "humble" if ros_version == "ros2" else "noetic")
        ros_setup = f"/opt/ros/{ros_distro}/setup.bash"

        full_command = f"source {ros_setup} && cd {workspace_path} && {cmd}"

        process = subprocess.run(
            ["bash", "-c", full_command],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for builds
            cwd=workspace_path
        )

        result["stdout"] = process.stdout
        result["stderr"] = process.stderr
        result["success"] = process.returncode == 0

        if not result["success"]:
            result["error"] = "Build failed. Check stderr for details."

    except subprocess.TimeoutExpired:
        result["error"] = "Build timed out after 5 minutes"
    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def check_ros_topics(
    ros_version: str = "ros2",
    workspace_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    List available ROS topics.

    Args:
        ros_version: Either "ros1" or "ros2"
        workspace_path: Optional workspace path for environment setup

    Returns:
        Dictionary with list of topics
    """
    if ros_version == "ros2":
        cmd = "ros2 topic list"
    else:
        cmd = "rostopic list"

    return run_ros_command.invoke({
        "command": cmd,
        "ros_version": ros_version,
        "workspace_path": workspace_path
    })


@tool
def check_ros_nodes(
    ros_version: str = "ros2",
    workspace_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    List running ROS nodes.

    Args:
        ros_version: Either "ros1" or "ros2"
        workspace_path: Optional workspace path for environment setup

    Returns:
        Dictionary with list of nodes
    """
    if ros_version == "ros2":
        cmd = "ros2 node list"
    else:
        cmd = "rosnode list"

    return run_ros_command.invoke({
        "command": cmd,
        "ros_version": ros_version,
        "workspace_path": workspace_path
    })


@tool
def get_topic_info(
    topic_name: str,
    ros_version: str = "ros2",
    workspace_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get information about a specific ROS topic.

    Args:
        topic_name: Name of the topic
        ros_version: Either "ros1" or "ros2"
        workspace_path: Optional workspace path

    Returns:
        Dictionary with topic information
    """
    if ros_version == "ros2":
        cmd = f"ros2 topic info {topic_name} --verbose"
    else:
        cmd = f"rostopic info {topic_name}"

    return run_ros_command.invoke({
        "command": cmd,
        "ros_version": ros_version,
        "workspace_path": workspace_path
    })


@tool
def get_message_type(
    message_type: str,
    ros_version: str = "ros2",
    workspace_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get the definition of a ROS message type.

    Args:
        message_type: Full message type (e.g., "geometry_msgs/msg/Twist")
        ros_version: Either "ros1" or "ros2"
        workspace_path: Optional workspace path

    Returns:
        Dictionary with message definition
    """
    if ros_version == "ros2":
        cmd = f"ros2 interface show {message_type}"
    else:
        cmd = f"rosmsg show {message_type}"

    return run_ros_command.invoke({
        "command": cmd,
        "ros_version": ros_version,
        "workspace_path": workspace_path
    })
