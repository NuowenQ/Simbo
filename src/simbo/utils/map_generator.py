"""
Map Generator Utilities for converting 2D images to Gazebo world files.

This module provides functions for:
- Edge detection in images
- Shape detection (rectangles, circles)
- Gazebo world file generation
"""

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from typing import List, Dict, Any, Tuple, Optional
import io
import base64
import math


def detect_edges(image_bytes: bytes) -> Dict[str, Any]:
    """
    Detect edges in an uploaded image using OpenCV.
    
    Args:
        image_bytes: Image file bytes
        
    Returns:
        Dictionary with detected edges, image data, and dimensions
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return {
            "success": False,
            "error": "OpenCV (cv2) and NumPy are required for edge detection. Please install: pip install opencv-python numpy"
        }
    
    try:
        # Read image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                "success": False,
                "error": "Invalid image file"
            }
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        # Convert detected lines to our format
        detected_edges = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                detected_edges.append({
                    "start": {"x": float(x1), "y": float(y1)},
                    "end": {"x": float(x2), "y": float(y2)}
                })
        
        # Convert image to base64 for display
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "edges": detected_edges,
            "image_data": f"data:image/png;base64,{img_base64}",
            "image_width": width,
            "image_height": height,
            "message": f"Detected {len(detected_edges)} edges"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Edge detection failed: {str(e)}"
        }


def detect_shapes(image_bytes: bytes) -> Dict[str, Any]:
    """
    Detect shapes (rectangles and circles) in an uploaded image using OpenCV.
    
    Args:
        image_bytes: Image file bytes
        
    Returns:
        Dictionary with detected shapes, lines, image data, and dimensions
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return {
            "success": False,
            "error": "OpenCV (cv2) and NumPy are required for shape detection. Please install: pip install opencv-python numpy"
        }
    
    try:
        # Read image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                "success": False,
                "error": "Invalid image file"
            }
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold for better shape detection
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Detect shapes using contours
        detected_shapes = []
        detected_lines = []
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            # Filter very small contours (noise)
            area = cv2.contourArea(contour)
            if area < 50:
                continue
            
            # Approximate contour to polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(approx)
            
            # Detect rectangles (4 corners)
            if len(approx) == 4:
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.2 <= aspect_ratio <= 5.0:
                    center_x = x + w / 2
                    center_y = y + h / 2
                    
                    # Check for duplicates
                    is_duplicate = False
                    for shape in detected_shapes:
                        if shape["type"] == "rectangle":
                            dist = math.sqrt(
                                (shape["center"]["x"] - center_x)**2 + 
                                (shape["center"]["y"] - center_y)**2
                            )
                            size_diff = abs(shape["width"] - w) + abs(shape["height"] - h)
                            if dist < 5 and size_diff < 10:
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        detected_shapes.append({
                            "type": "rectangle",
                            "center": {"x": float(center_x), "y": float(center_y)},
                            "width": float(w),
                            "height": float(h),
                            "selected": True
                        })
                    continue
            
            # Detect circles using circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            if circularity > 0.8:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    radius = math.sqrt(area / math.pi)
                    
                    if radius >= 5:
                        # Check for duplicates
                        is_duplicate = False
                        for shape in detected_shapes:
                            if shape["type"] == "circle":
                                dist = math.sqrt(
                                    (shape["center"]["x"] - cx)**2 + 
                                    (shape["center"]["y"] - cy)**2
                                )
                                radius_diff = abs(shape["radius"] - radius)
                                if dist < 3 and radius_diff < 3:
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            detected_shapes.append({
                                "type": "circle",
                                "center": {"x": float(cx), "y": float(cy)},
                                "radius": float(radius),
                                "selected": True
                            })
                continue
            
            # If not a clear shape, convert to lines
            if len(approx) >= 3:
                for i in range(len(approx)):
                    pt1 = approx[i][0]
                    pt2 = approx[(i + 1) % len(approx)][0]
                    detected_lines.append({
                        "start": {"x": float(pt1[0]), "y": float(pt1[1])},
                        "end": {"x": float(pt2[0]), "y": float(pt2[1])},
                        "selected": True
                    })
        
        # Also detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                
                # Check for duplicates
                is_duplicate = False
                for shape in detected_shapes:
                    if shape["type"] == "circle":
                        dist = math.sqrt(
                            (shape["center"]["x"] - x)**2 + 
                            (shape["center"]["y"] - y)**2
                        )
                        radius_diff = abs(shape["radius"] - r)
                        if dist < 5 and radius_diff < 5:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    detected_shapes.append({
                        "type": "circle",
                        "center": {"x": float(x), "y": float(y)},
                        "radius": float(r),
                        "selected": True
                    })
        
        # Convert image to base64 for display
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "shapes": detected_shapes,
            "lines": detected_lines,
            "image_data": f"data:image/png;base64,{img_base64}",
            "image_width": width,
            "image_height": height,
            "message": f"Detected {len(detected_shapes)} shapes and {len(detected_lines)} lines"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Shape detection failed: {str(e)}"
        }


def generate_gazebo_world(
    walls: List[Dict[str, Any]],
    rooms: List[Dict[str, Any]],
    scale: float = 20.0,
    wall_height: float = 3.0,
    wall_thickness: float = 0.2
) -> str:
    """
    Generate Gazebo world file content from map data.
    
    Args:
        walls: List of wall dictionaries with 'start' and 'end' points
        rooms: List of room dictionaries with 'points' list
        scale: Pixels per meter (default: 20)
        wall_height: Height of walls in meters (default: 3.0)
        wall_thickness: Thickness of walls in meters (default: 0.2)
        
    Returns:
        Gazebo world file content as string
    """
    world_content = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="generated_world">

    <!-- Physics -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

"""
    
    # Generate walls
    for i, wall in enumerate(walls):
        # Convert pixel coordinates to meters
        x1 = wall['start']['x'] / scale
        y1 = wall['start']['y'] / scale
        x2 = wall['end']['x'] / scale
        y2 = wall['end']['y'] / scale
        
        # Calculate wall center, length and angle
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = math.atan2(y2 - y1, x2 - x1)
        
        wall_content = f"""
    <!-- Wall {i} -->
    <model name="wall_{i}">
      <static>true</static>
      <pose>{center_x} {center_y} {wall_height/2} 0 0 {angle}</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>{length} {wall_thickness} {wall_height}</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>{length} {wall_thickness} {wall_height}</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""
        world_content += wall_content
    
    # Generate floor for rooms
    for i, room in enumerate(rooms):
        if len(room.get('points', [])) < 3:
            continue
        
        points = room['points']
        # Calculate room center
        center_x = sum(p['x'] for p in points) / len(points) / scale
        center_y = sum(p['y'] for p in points) / len(points) / scale
        
        # Estimate room size (simple bounding box)
        min_x = min(p['x'] for p in points) / scale
        max_x = max(p['x'] for p in points) / scale
        min_y = min(p['y'] for p in points) / scale
        max_y = max(p['y'] for p in points) / scale
        
        size_x = max_x - min_x
        size_y = max_y - min_y
        
        room_content = f"""
    <!-- Room {i}: {room.get('name', 'room')} -->
    <model name="room_{i}_floor">
      <static>true</static>
      <pose>{center_x} {center_y} 0.001 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box>
              <size>{size_x} {size_y} 0.001</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.9 1.0 1</ambient>
            <diffuse>0.8 0.9 1.0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""
        world_content += room_content
    
    world_content += """
  </world>
</sdf>
"""
    
    return world_content
