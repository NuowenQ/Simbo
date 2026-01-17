"""Simbo Tools Module"""

from .workspace_tools import (
    detect_ros_version,
    analyze_workspace,
    list_packages,
    read_package_xml,
    find_launch_files,
    find_source_files,
)
from .file_tools import (
    read_file,
    write_file,
    edit_file,
    insert_at_line,
    delete_lines,
    create_directory,
    list_directory,
    search_in_files,
    copy_file,
    delete_file,
)
from .shell_tools import (
    run_command,
    run_ros_command,
    build_ros_workspace,
    check_ros_topics,
    check_ros_nodes,
    get_topic_info,
    get_message_type,
)
from .code_tools import (
    search_code,
    generate_controller,
)

__all__ = [
    # Workspace tools
    "detect_ros_version",
    "analyze_workspace",
    "list_packages",
    "read_package_xml",
    "find_launch_files",
    "find_source_files",
    # File tools
    "read_file",
    "write_file",
    "edit_file",
    "insert_at_line",
    "delete_lines",
    "create_directory",
    "list_directory",
    "search_in_files",
    "copy_file",
    "delete_file",
    # Shell tools
    "run_command",
    "run_ros_command",
    "build_ros_workspace",
    "check_ros_topics",
    "check_ros_nodes",
    "get_topic_info",
    "get_message_type",
    # Code tools
    "search_code",
    "generate_controller",
]
