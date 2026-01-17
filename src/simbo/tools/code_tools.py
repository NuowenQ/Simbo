"""
Tools for code manipulation and generation in ROS workspaces.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.tools import tool


@tool
def read_file(file_path: str) -> Dict[str, any]:
    """
    Read the contents of a file.

    Args:
        file_path: Path to the file to read

    Returns:
        Dictionary with file content and metadata
    """
    result = {
        "exists": False,
        "path": file_path,
        "content": "",
        "lines": 0,
        "size": 0,
        "extension": "",
        "error": None
    }

    if not os.path.exists(file_path):
        result["error"] = f"File not found: {file_path}"
        return result

    if not os.path.isfile(file_path):
        result["error"] = f"Path is not a file: {file_path}"
        return result

    result["exists"] = True
    result["extension"] = os.path.splitext(file_path)[1]
    result["size"] = os.path.getsize(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            result["content"] = content
            result["lines"] = len(content.splitlines())
    except UnicodeDecodeError:
        result["error"] = "File appears to be binary"
    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def write_file(file_path: str, content: str, create_dirs: bool = True) -> Dict[str, any]:
    """
    Write content to a file.

    Args:
        file_path: Path to the file to write
        content: Content to write
        create_dirs: Whether to create parent directories if they don't exist

    Returns:
        Dictionary with operation result
    """
    result = {
        "success": False,
        "path": file_path,
        "bytes_written": 0,
        "error": None
    }

    try:
        if create_dirs:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            bytes_written = f.write(content)
            result["bytes_written"] = bytes_written
            result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


@tool
def search_code(
    workspace_path: str,
    pattern: str,
    file_extensions: Optional[List[str]] = None,
    case_sensitive: bool = False
) -> List[Dict[str, any]]:
    """
    Search for a pattern in code files.

    Args:
        workspace_path: Path to search in
        pattern: Regex pattern to search for
        file_extensions: List of extensions to search (e.g., ['.py', '.cpp'])
        case_sensitive: Whether search should be case sensitive

    Returns:
        List of matches with file path, line number, and content
    """
    matches = []
    flags = 0 if case_sensitive else re.IGNORECASE

    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return [{"error": f"Invalid regex pattern: {e}"}]

    if file_extensions is None:
        file_extensions = [".py", ".cpp", ".c", ".h", ".hpp", ".xml", ".yaml", ".yml", ".launch"]

    for root, dirs, files in os.walk(workspace_path):
        # Skip common non-source directories
        dirs[:] = [d for d in dirs if d not in [
            "build", "install", "log", "__pycache__", ".git", "node_modules", ".venv"
        ]]

        for filename in files:
            if any(filename.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                matches.append({
                                    "file": file_path,
                                    "line_number": line_num,
                                    "content": line.strip(),
                                    "match": regex.findall(line)
                                })
                except (UnicodeDecodeError, IOError):
                    continue

    return matches


@tool
def generate_controller(
    controller_type: str,
    robot_name: str,
    ros_version: str,
    additional_params: Optional[Dict[str, any]] = None
) -> Dict[str, str]:
    """
    Generate a robot controller based on specified parameters.

    Args:
        controller_type: Type of controller (e.g., 'velocity', 'position', 'twist', 'joint_trajectory')
        robot_name: Name of the robot
        ros_version: ROS version ('ros1' or 'ros2')
        additional_params: Additional parameters for customization

    Returns:
        Dictionary with generated code and metadata
    """
    params = additional_params or {}

    if ros_version == "ros2":
        return _generate_ros2_controller(controller_type, robot_name, params)
    else:
        return _generate_ros1_controller(controller_type, robot_name, params)


def _generate_ros2_controller(
    controller_type: str,
    robot_name: str,
    params: Dict[str, any]
) -> Dict[str, str]:
    """Generate ROS2 controller code."""

    templates = {
        "velocity": '''#!/usr/bin/env python3
"""
Velocity controller for {robot_name}.
Generated by Simbo - ROS2 Gazebo Simulation Assistant.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import math


class VelocityController(Node):
    """Velocity controller node for {robot_name}."""

    def __init__(self):
        super().__init__('{robot_name_lower}_velocity_controller')

        # Parameters
        self.declare_parameter('linear_speed', {linear_speed})
        self.declare_parameter('angular_speed', {angular_speed})
        self.declare_parameter('publish_rate', {publish_rate})

        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.publish_rate = self.get_parameter('publish_rate').value

        # Publisher for velocity commands
        self.vel_publisher = self.create_publisher(
            Twist,
            '/{robot_name_lower}/cmd_vel',
            10
        )

        # Subscriber for velocity input (optional external control)
        self.vel_subscriber = self.create_subscription(
            Twist,
            '/{robot_name_lower}/cmd_vel_input',
            self.velocity_callback,
            10
        )

        # Timer for periodic publishing
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.timer_callback
        )

        # Current velocity command
        self.current_twist = Twist()

        self.get_logger().info(f'{{self.get_name()}} initialized')

    def velocity_callback(self, msg: Twist):
        """Handle incoming velocity commands."""
        self.current_twist = msg

    def timer_callback(self):
        """Publish current velocity command."""
        self.vel_publisher.publish(self.current_twist)

    def set_velocity(self, linear_x: float, angular_z: float):
        """Set the velocity command."""
        self.current_twist.linear.x = linear_x
        self.current_twist.angular.z = angular_z

    def stop(self):
        """Stop the robot."""
        self.current_twist = Twist()
        self.vel_publisher.publish(self.current_twist)


def main(args=None):
    rclpy.init(args=args)
    controller = VelocityController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down...')
    finally:
        controller.stop()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
''',

        "position": '''#!/usr/bin/env python3
"""
Position controller for {robot_name}.
Generated by Simbo - ROS2 Gazebo Simulation Assistant.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import math
import numpy as np


class PositionController(Node):
    """Position controller node for {robot_name}."""

    def __init__(self):
        super().__init__('{robot_name_lower}_position_controller')

        # Parameters
        self.declare_parameter('kp_linear', {kp_linear})
        self.declare_parameter('kp_angular', {kp_angular})
        self.declare_parameter('goal_tolerance', {goal_tolerance})
        self.declare_parameter('max_linear_vel', {max_linear_vel})
        self.declare_parameter('max_angular_vel', {max_angular_vel})

        self.kp_linear = self.get_parameter('kp_linear').value
        self.kp_angular = self.get_parameter('kp_angular').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value

        # Publishers
        self.vel_publisher = self.create_publisher(
            Twist,
            '/{robot_name_lower}/cmd_vel',
            10
        )

        # Subscribers
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/{robot_name_lower}/odom',
            self.odom_callback,
            10
        )

        self.goal_subscriber = self.create_subscription(
            PoseStamped,
            '/{robot_name_lower}/goal_pose',
            self.goal_callback,
            10
        )

        # State
        self.current_pose = None
        self.goal_pose = None
        self.goal_reached = True

        # Control loop timer
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(f'{{self.get_name()}} initialized')

    def odom_callback(self, msg: Odometry):
        """Update current pose from odometry."""
        self.current_pose = msg.pose.pose

    def goal_callback(self, msg: PoseStamped):
        """Set new goal pose."""
        self.goal_pose = msg.pose
        self.goal_reached = False
        self.get_logger().info(
            f'New goal: x={{self.goal_pose.position.x:.2f}}, '
            f'y={{self.goal_pose.position.y:.2f}}'
        )

    def control_loop(self):
        """Main control loop."""
        if self.current_pose is None or self.goal_pose is None or self.goal_reached:
            return

        # Calculate error
        dx = self.goal_pose.position.x - self.current_pose.position.x
        dy = self.goal_pose.position.y - self.current_pose.position.y
        distance = math.sqrt(dx**2 + dy**2)

        # Check if goal reached
        if distance < self.goal_tolerance:
            self.goal_reached = True
            self.vel_publisher.publish(Twist())
            self.get_logger().info('Goal reached!')
            return

        # Calculate desired heading
        desired_yaw = math.atan2(dy, dx)
        current_yaw = self._get_yaw_from_quaternion(self.current_pose.orientation)
        yaw_error = self._normalize_angle(desired_yaw - current_yaw)

        # Calculate control commands
        cmd = Twist()
        cmd.linear.x = min(self.kp_linear * distance, self.max_linear_vel)
        cmd.angular.z = max(min(self.kp_angular * yaw_error, self.max_angular_vel), -self.max_angular_vel)

        # Reduce linear velocity when turning
        if abs(yaw_error) > 0.5:
            cmd.linear.x *= 0.5

        self.vel_publisher.publish(cmd)

    def _get_yaw_from_quaternion(self, q):
        """Extract yaw from quaternion."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    controller = PositionController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down...')
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
''',

        "joint_trajectory": '''#!/usr/bin/env python3
"""
Joint trajectory controller for {robot_name}.
Generated by Simbo - ROS2 Gazebo Simulation Assistant.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import numpy as np


class JointTrajectoryController(Node):
    """Joint trajectory controller for {robot_name}."""

    def __init__(self):
        super().__init__('{robot_name_lower}_joint_controller')

        # Parameters
        self.declare_parameter('joint_names', {joint_names})
        self.declare_parameter('action_server', '/{robot_name_lower}/joint_trajectory_controller/follow_joint_trajectory')

        self.joint_names = self.get_parameter('joint_names').value
        action_server = self.get_parameter('action_server').value

        # Action client for trajectory execution
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            action_server
        )

        # Subscriber for current joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/{robot_name_lower}/joint_states',
            self.joint_state_callback,
            10
        )

        # Current joint positions
        self.current_positions = {{}}

        self.get_logger().info(f'{{self.get_name()}} initialized')
        self.get_logger().info(f'Controlling joints: {{self.joint_names}}')

    def joint_state_callback(self, msg: JointState):
        """Update current joint positions."""
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                self.current_positions[name] = msg.position[i]

    def send_trajectory(self, positions: list, duration: float = 2.0):
        """
        Send a trajectory to move joints to specified positions.

        Args:
            positions: List of target joint positions
            duration: Time to complete the motion in seconds
        """
        if len(positions) != len(self.joint_names):
            self.get_logger().error(
                f'Position count ({{len(positions)}}) does not match '
                f'joint count ({{len(self.joint_names)}})'
            )
            return

        # Wait for action server
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available!')
            return

        # Create trajectory message
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        trajectory.points.append(point)

        # Create goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        # Send goal
        self.get_logger().info(f'Sending trajectory to positions: {{positions}}')
        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        """Handle goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected!')
            return

        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        """Handle result."""
        result = future.result().result
        self.get_logger().info(f'Trajectory execution completed')

    def move_to_home(self):
        """Move all joints to home position (zeros)."""
        home_positions = [0.0] * len(self.joint_names)
        self.send_trajectory(home_positions, duration=3.0)


def main(args=None):
    rclpy.init(args=args)
    controller = JointTrajectoryController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down...')
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
''',

        "twist": '''#!/usr/bin/env python3
"""
Twist teleop controller for {robot_name}.
Generated by Simbo - ROS2 Gazebo Simulation Assistant.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import select


class TwistTeleop(Node):
    """Keyboard teleop controller for {robot_name}."""

    INSTRUCTIONS = """
    Control {robot_name}!
    ---------------------------
    Moving:
        w
    a   s   d

    w/s : increase/decrease linear velocity
    a/d : increase/decrease angular velocity
    space : stop

    q : quit
    """

    def __init__(self):
        super().__init__('{robot_name_lower}_teleop')

        # Parameters
        self.declare_parameter('linear_step', {linear_step})
        self.declare_parameter('angular_step', {angular_step})
        self.declare_parameter('max_linear', {max_linear})
        self.declare_parameter('max_angular', {max_angular})

        self.linear_step = self.get_parameter('linear_step').value
        self.angular_step = self.get_parameter('angular_step').value
        self.max_linear = self.get_parameter('max_linear').value
        self.max_angular = self.get_parameter('max_angular').value

        # Publisher
        self.publisher = self.create_publisher(
            Twist,
            '/{robot_name_lower}/cmd_vel',
            10
        )

        # Current velocity
        self.linear_vel = 0.0
        self.angular_vel = 0.0

        # Store terminal settings
        self.settings = termios.tcgetattr(sys.stdin)

        self.get_logger().info(self.INSTRUCTIONS)

    def get_key(self):
        """Get keyboard input."""
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run(self):
        """Main teleop loop."""
        try:
            while True:
                key = self.get_key()

                if key == 'w':
                    self.linear_vel = min(self.linear_vel + self.linear_step, self.max_linear)
                elif key == 's':
                    self.linear_vel = max(self.linear_vel - self.linear_step, -self.max_linear)
                elif key == 'a':
                    self.angular_vel = min(self.angular_vel + self.angular_step, self.max_angular)
                elif key == 'd':
                    self.angular_vel = max(self.angular_vel - self.angular_step, -self.max_angular)
                elif key == ' ':
                    self.linear_vel = 0.0
                    self.angular_vel = 0.0
                elif key == 'q':
                    break

                # Publish twist
                twist = Twist()
                twist.linear.x = self.linear_vel
                twist.angular.z = self.angular_vel
                self.publisher.publish(twist)

                print(f'Linear: {{self.linear_vel:.2f}}, Angular: {{self.angular_vel:.2f}}', end='\\r')

        except Exception as e:
            self.get_logger().error(f'Error: {{e}}')
        finally:
            # Stop robot
            self.publisher.publish(Twist())
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main(args=None):
    rclpy.init(args=args)
    teleop = TwistTeleop()

    try:
        teleop.run()
    except KeyboardInterrupt:
        pass
    finally:
        teleop.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
'''
    }

    # Default parameters
    defaults = {
        "linear_speed": params.get("linear_speed", 0.5),
        "angular_speed": params.get("angular_speed", 1.0),
        "publish_rate": params.get("publish_rate", 10.0),
        "kp_linear": params.get("kp_linear", 0.5),
        "kp_angular": params.get("kp_angular", 1.0),
        "goal_tolerance": params.get("goal_tolerance", 0.1),
        "max_linear_vel": params.get("max_linear_vel", 0.5),
        "max_angular_vel": params.get("max_angular_vel", 1.0),
        "joint_names": params.get("joint_names", "['joint1', 'joint2', 'joint3']"),
        "linear_step": params.get("linear_step", 0.1),
        "angular_step": params.get("angular_step", 0.1),
        "max_linear": params.get("max_linear", 1.0),
        "max_angular": params.get("max_angular", 2.0),
        "robot_name": robot_name,
        "robot_name_lower": robot_name.lower().replace(" ", "_").replace("-", "_")
    }

    template = templates.get(controller_type, templates["velocity"])
    code = template.format(**defaults)

    return {
        "code": code,
        "controller_type": controller_type,
        "robot_name": robot_name,
        "ros_version": "ros2",
        "filename": f"{defaults['robot_name_lower']}_{controller_type}_controller.py",
        "description": f"{controller_type.replace('_', ' ').title()} controller for {robot_name}"
    }


def _generate_ros1_controller(
    controller_type: str,
    robot_name: str,
    params: Dict[str, any]
) -> Dict[str, str]:
    """Generate ROS1 controller code."""

    templates = {
        "velocity": '''#!/usr/bin/env python
"""
Velocity controller for {robot_name}.
Generated by Simbo - ROS1 Gazebo Simulation Assistant.
"""

import rospy
from geometry_msgs.msg import Twist


class VelocityController:
    """Velocity controller for {robot_name}."""

    def __init__(self):
        rospy.init_node('{robot_name_lower}_velocity_controller')

        # Parameters
        self.linear_speed = rospy.get_param('~linear_speed', {linear_speed})
        self.angular_speed = rospy.get_param('~angular_speed', {angular_speed})
        self.publish_rate = rospy.get_param('~publish_rate', {publish_rate})

        # Publisher
        self.vel_pub = rospy.Publisher(
            '/{robot_name_lower}/cmd_vel',
            Twist,
            queue_size=10
        )

        # Subscriber
        self.vel_sub = rospy.Subscriber(
            '/{robot_name_lower}/cmd_vel_input',
            Twist,
            self.velocity_callback
        )

        # Current command
        self.current_twist = Twist()

        # Timer
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate),
            self.timer_callback
        )

        rospy.loginfo('{robot_name} velocity controller initialized')

    def velocity_callback(self, msg):
        """Handle incoming velocity commands."""
        self.current_twist = msg

    def timer_callback(self, event):
        """Publish current velocity."""
        self.vel_pub.publish(self.current_twist)

    def set_velocity(self, linear_x, angular_z):
        """Set velocity command."""
        self.current_twist.linear.x = linear_x
        self.current_twist.angular.z = angular_z

    def stop(self):
        """Stop the robot."""
        self.current_twist = Twist()
        self.vel_pub.publish(self.current_twist)


def main():
    controller = VelocityController()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        controller.stop()


if __name__ == '__main__':
    main()
''',

        "position": '''#!/usr/bin/env python
"""
Position controller for {robot_name}.
Generated by Simbo - ROS1 Gazebo Simulation Assistant.
"""

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import math


class PositionController:
    """Position controller for {robot_name}."""

    def __init__(self):
        rospy.init_node('{robot_name_lower}_position_controller')

        # Parameters
        self.kp_linear = rospy.get_param('~kp_linear', {kp_linear})
        self.kp_angular = rospy.get_param('~kp_angular', {kp_angular})
        self.goal_tolerance = rospy.get_param('~goal_tolerance', {goal_tolerance})
        self.max_linear_vel = rospy.get_param('~max_linear_vel', {max_linear_vel})
        self.max_angular_vel = rospy.get_param('~max_angular_vel', {max_angular_vel})

        # Publisher
        self.vel_pub = rospy.Publisher(
            '/{robot_name_lower}/cmd_vel',
            Twist,
            queue_size=10
        )

        # Subscribers
        self.odom_sub = rospy.Subscriber(
            '/{robot_name_lower}/odom',
            Odometry,
            self.odom_callback
        )

        self.goal_sub = rospy.Subscriber(
            '/{robot_name_lower}/goal_pose',
            PoseStamped,
            self.goal_callback
        )

        # State
        self.current_pose = None
        self.goal_pose = None
        self.goal_reached = True

        # Control loop
        self.rate = rospy.Rate(20)

        rospy.loginfo('{robot_name} position controller initialized')

    def odom_callback(self, msg):
        """Update current pose."""
        self.current_pose = msg.pose.pose

    def goal_callback(self, msg):
        """Set new goal."""
        self.goal_pose = msg.pose
        self.goal_reached = False
        rospy.loginfo('New goal: x=%.2f, y=%.2f' %
                      (self.goal_pose.position.x, self.goal_pose.position.y))

    def control_loop(self):
        """Main control loop."""
        while not rospy.is_shutdown():
            if self.current_pose and self.goal_pose and not self.goal_reached:
                dx = self.goal_pose.position.x - self.current_pose.position.x
                dy = self.goal_pose.position.y - self.current_pose.position.y
                distance = math.sqrt(dx**2 + dy**2)

                if distance < self.goal_tolerance:
                    self.goal_reached = True
                    self.vel_pub.publish(Twist())
                    rospy.loginfo('Goal reached!')
                else:
                    desired_yaw = math.atan2(dy, dx)
                    current_yaw = self._get_yaw(self.current_pose.orientation)
                    yaw_error = self._normalize_angle(desired_yaw - current_yaw)

                    cmd = Twist()
                    cmd.linear.x = min(self.kp_linear * distance, self.max_linear_vel)
                    cmd.angular.z = max(min(self.kp_angular * yaw_error,
                                           self.max_angular_vel), -self.max_angular_vel)

                    if abs(yaw_error) > 0.5:
                        cmd.linear.x *= 0.5

                    self.vel_pub.publish(cmd)

            self.rate.sleep()

    def _get_yaw(self, q):
        """Get yaw from quaternion."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def main():
    controller = PositionController()
    try:
        controller.control_loop()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
'''
    }

    defaults = {
        "linear_speed": params.get("linear_speed", 0.5),
        "angular_speed": params.get("angular_speed", 1.0),
        "publish_rate": params.get("publish_rate", 10.0),
        "kp_linear": params.get("kp_linear", 0.5),
        "kp_angular": params.get("kp_angular", 1.0),
        "goal_tolerance": params.get("goal_tolerance", 0.1),
        "max_linear_vel": params.get("max_linear_vel", 0.5),
        "max_angular_vel": params.get("max_angular_vel", 1.0),
        "robot_name": robot_name,
        "robot_name_lower": robot_name.lower().replace(" ", "_").replace("-", "_")
    }

    template = templates.get(controller_type, templates["velocity"])
    code = template.format(**defaults)

    return {
        "code": code,
        "controller_type": controller_type,
        "robot_name": robot_name,
        "ros_version": "ros1",
        "filename": f"{defaults['robot_name_lower']}_{controller_type}_controller.py",
        "description": f"{controller_type.replace('_', ' ').title()} controller for {robot_name}"
    }
