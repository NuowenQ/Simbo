"""
File manipulation tools for autonomous code editing.
These tools allow the agent to directly modify the user's codebase.
"""

import os
import shutil
import difflib
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain_core.tools import tool


@tool
def read_file(file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> Dict[str, Any]:
    """
    Read the contents of a file. Use this to understand existing code before making changes.

    Args:
        file_path: Absolute path to the file to read
        start_line: Optional starting line number (1-indexed)
        end_line: Optional ending line number (1-indexed)

    Returns:
        Dictionary with file content, line count, and metadata
    """
    result = {
        "success": False,
        "path": file_path,
        "content": "",
        "lines": [],
        "total_lines": 0,
        "error": None
    }

    if not os.path.exists(file_path):
        result["error"] = f"File not found: {file_path}"
        return result

    if not os.path.isfile(file_path):
        result["error"] = f"Path is not a file: {file_path}"
        return result

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()

        result["total_lines"] = len(all_lines)

        if start_line is not None or end_line is not None:
            start = (start_line - 1) if start_line else 0
            end = end_line if end_line else len(all_lines)
            selected_lines = all_lines[start:end]
        else:
            selected_lines = all_lines

        # Format with line numbers
        if start_line:
            numbered_lines = [f"{i + start_line}: {line.rstrip()}" for i, line in enumerate(selected_lines)]
        else:
            numbered_lines = [f"{i + 1}: {line.rstrip()}" for i, line in enumerate(selected_lines)]

        result["content"] = "\n".join(numbered_lines)
        result["lines"] = [line.rstrip() for line in selected_lines]
        result["success"] = True

    except UnicodeDecodeError:
        result["error"] = "File appears to be binary and cannot be read as text"
    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def write_file(file_path: str, content: str) -> Dict[str, Any]:
    """
    Write content to a file, creating it if it doesn't exist.
    This will OVERWRITE the entire file. Use edit_file for partial changes.

    Args:
        file_path: Absolute path to the file to write
        content: Complete content to write to the file

    Returns:
        Dictionary with operation result
    """
    result = {
        "success": False,
        "path": file_path,
        "created": False,
        "bytes_written": 0,
        "error": None
    }

    try:
        # Create parent directories if needed
        parent_dir = os.path.dirname(file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        created = not os.path.exists(file_path)

        with open(file_path, "w", encoding="utf-8") as f:
            bytes_written = f.write(content)

        result["success"] = True
        result["created"] = created
        result["bytes_written"] = bytes_written

    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def edit_file(
    file_path: str,
    old_content: str,
    new_content: str
) -> Dict[str, Any]:
    """
    Edit a file by replacing specific content. This is safer than write_file
    for making targeted changes to existing files.

    Args:
        file_path: Absolute path to the file to edit
        old_content: The exact content to find and replace (must match exactly)
        new_content: The new content to replace it with

    Returns:
        Dictionary with operation result and diff
    """
    result = {
        "success": False,
        "path": file_path,
        "replacements": 0,
        "diff": "",
        "error": None
    }

    if not os.path.exists(file_path):
        result["error"] = f"File not found: {file_path}"
        return result

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        if old_content not in original_content:
            result["error"] = "Could not find the specified content to replace. Make sure old_content matches exactly."
            return result

        # Count replacements
        replacements = original_content.count(old_content)
        new_file_content = original_content.replace(old_content, new_content)

        # Generate diff
        diff = difflib.unified_diff(
            original_content.splitlines(keepends=True),
            new_file_content.splitlines(keepends=True),
            fromfile=f"a/{os.path.basename(file_path)}",
            tofile=f"b/{os.path.basename(file_path)}"
        )
        result["diff"] = "".join(diff)

        # Write the changes
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_file_content)

        result["success"] = True
        result["replacements"] = replacements

    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def insert_at_line(
    file_path: str,
    line_number: int,
    content: str
) -> Dict[str, Any]:
    """
    Insert content at a specific line number in a file.

    Args:
        file_path: Absolute path to the file
        line_number: Line number to insert at (1-indexed, content inserted before this line)
        content: Content to insert

    Returns:
        Dictionary with operation result
    """
    result = {
        "success": False,
        "path": file_path,
        "error": None
    }

    if not os.path.exists(file_path):
        result["error"] = f"File not found: {file_path}"
        return result

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Adjust for 1-indexed line numbers
        insert_index = max(0, min(line_number - 1, len(lines)))

        # Ensure content ends with newline
        if not content.endswith("\n"):
            content += "\n"

        lines.insert(insert_index, content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def delete_lines(
    file_path: str,
    start_line: int,
    end_line: int
) -> Dict[str, Any]:
    """
    Delete a range of lines from a file.

    Args:
        file_path: Absolute path to the file
        start_line: Starting line number (1-indexed, inclusive)
        end_line: Ending line number (1-indexed, inclusive)

    Returns:
        Dictionary with operation result
    """
    result = {
        "success": False,
        "path": file_path,
        "lines_deleted": 0,
        "error": None
    }

    if not os.path.exists(file_path):
        result["error"] = f"File not found: {file_path}"
        return result

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Adjust for 1-indexed line numbers
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)

        lines_deleted = end_idx - start_idx
        new_lines = lines[:start_idx] + lines[end_idx:]

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        result["success"] = True
        result["lines_deleted"] = lines_deleted

    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def create_directory(dir_path: str) -> Dict[str, Any]:
    """
    Create a directory and any necessary parent directories.

    Args:
        dir_path: Absolute path to the directory to create

    Returns:
        Dictionary with operation result
    """
    result = {
        "success": False,
        "path": dir_path,
        "created": False,
        "error": None
    }

    try:
        if os.path.exists(dir_path):
            result["success"] = True
            result["created"] = False
            return result

        os.makedirs(dir_path, exist_ok=True)
        result["success"] = True
        result["created"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def list_directory(dir_path: str, recursive: bool = False) -> Dict[str, Any]:
    """
    List contents of a directory.

    Args:
        dir_path: Absolute path to the directory
        recursive: Whether to list recursively

    Returns:
        Dictionary with directory contents
    """
    result = {
        "success": False,
        "path": dir_path,
        "files": [],
        "directories": [],
        "error": None
    }

    if not os.path.exists(dir_path):
        result["error"] = f"Directory not found: {dir_path}"
        return result

    if not os.path.isdir(dir_path):
        result["error"] = f"Path is not a directory: {dir_path}"
        return result

    try:
        if recursive:
            for root, dirs, files in os.walk(dir_path):
                # Skip hidden and build directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['build', 'install', 'log', '__pycache__']]
                for f in files:
                    if not f.startswith('.'):
                        result["files"].append(os.path.join(root, f))
                for d in dirs:
                    result["directories"].append(os.path.join(root, d))
        else:
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path):
                    result["files"].append(item_path)
                elif os.path.isdir(item_path):
                    result["directories"].append(item_path)

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def search_in_files(
    directory: str,
    pattern: str,
    file_extensions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Search for a pattern across multiple files in a directory.

    Args:
        directory: Directory to search in
        pattern: Text pattern to search for (case-insensitive)
        file_extensions: List of file extensions to search (e.g., ['.py', '.cpp'])

    Returns:
        Dictionary with search results
    """
    import re

    result = {
        "success": False,
        "matches": [],
        "total_matches": 0,
        "error": None
    }

    if not os.path.exists(directory):
        result["error"] = f"Directory not found: {directory}"
        return result

    if file_extensions is None:
        file_extensions = ['.py', '.cpp', '.c', '.h', '.hpp', '.xml', '.yaml', '.yml', '.launch', '.urdf', '.xacro', '.sdf']

    try:
        regex = re.compile(pattern, re.IGNORECASE)

        for root, dirs, files in os.walk(directory):
            # Skip hidden and build directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['build', 'install', 'log', '__pycache__', '.git']]

            for filename in files:
                if any(filename.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                if regex.search(line):
                                    result["matches"].append({
                                        "file": file_path,
                                        "line": line_num,
                                        "content": line.strip()[:200]  # Truncate long lines
                                    })
                    except (UnicodeDecodeError, IOError):
                        continue

        result["success"] = True
        result["total_matches"] = len(result["matches"])

    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def copy_file(source: str, destination: str) -> Dict[str, Any]:
    """
    Copy a file from source to destination.

    Args:
        source: Source file path
        destination: Destination file path

    Returns:
        Dictionary with operation result
    """
    result = {
        "success": False,
        "source": source,
        "destination": destination,
        "error": None
    }

    if not os.path.exists(source):
        result["error"] = f"Source file not found: {source}"
        return result

    try:
        # Create parent directories if needed
        parent_dir = os.path.dirname(destination)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        shutil.copy2(source, destination)
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def delete_file(file_path: str) -> Dict[str, Any]:
    """
    Delete a file.

    Args:
        file_path: Path to the file to delete

    Returns:
        Dictionary with operation result
    """
    result = {
        "success": False,
        "path": file_path,
        "error": None
    }

    if not os.path.exists(file_path):
        result["error"] = f"File not found: {file_path}"
        return result

    try:
        os.remove(file_path)
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result
