"""
ADK Tools for WAVEGO Pro Quadruped Robot.
Wraps BaseRobot hardware and simulation calls into typed LLM-callable functions.
"""

import base64
import math
import time
from typing import Any, Dict, Optional
from robot_interface import BaseRobot
from app.config import config

# Global robot instance reference set during initialization
_ROBOT_INSTANCE: Optional[BaseRobot] = None


def set_robot_instance(robot: BaseRobot):
    """Binds the active robot controller (Simulation or Physical) to the tools."""
    global _ROBOT_INSTANCE
    _ROBOT_INSTANCE = robot


def get_robot() -> BaseRobot:
    if _ROBOT_INSTANCE is None:
        raise RuntimeError("Robot instance has not been bound to tools.")
    return _ROBOT_INSTANCE


# ---------------------------------------------------------------------------
# Core Locomotion Helper
# ---------------------------------------------------------------------------

def walk_relative(distance_meters: float = 0.0, turn_degrees: float = 0.0) -> str:
    """
    Internal core controller for walking and turning with closed-loop telemetry.
    Positive distance moves forward; negative distance moves backward.
    Positive degrees turns right; negative degrees turns left.
    """
    robot = get_robot()
    start_pos, _ = robot.get_pose()
    start_rpy = robot.get_imu()
    initial_yaw = start_rpy[2]

    # Handle turning first
    if abs(turn_degrees) > 1.0:
        turn_dir = 35.0 if turn_degrees > 0 else -35.0
        robot.walk(0.0, turn_dir)
        target_yaw_diff = abs(turn_degrees)
        t_start = time.time()

        while (time.time() - t_start) < 25.0:
            if hasattr(robot, "step") and not getattr(robot, "has_active_viewer", False):
                robot.step()
            time.sleep(0.002)

            curr_yaw = robot.get_imu()[2]
            net_diff = curr_yaw - initial_yaw
            while net_diff > 180.0:
                net_diff -= 360.0
            while net_diff < -180.0:
                net_diff += 360.0

            if abs(net_diff) >= target_yaw_diff:
                break

        robot.stand()
        if not getattr(robot, "has_active_viewer", False):
            for _ in range(50):
                if hasattr(robot, "step"):
                    robot.step()

    # Handle translation next
    if abs(distance_meters) > 0.01:
        speed = 50.0 if distance_meters > 0 else -50.0
        robot.walk(speed, 0.0)
        target_dist = abs(distance_meters)
        t_start = time.time()

        while (time.time() - t_start) < 25.0:
            if hasattr(robot, "step") and not getattr(robot, "has_active_viewer", False):
                robot.step()
            time.sleep(0.002)
            curr_pos, _ = robot.get_pose()
            traveled = math.hypot(curr_pos[0] - start_pos[0], curr_pos[1] - start_pos[1])
            if traveled >= target_dist:
                break

        robot.stand()
        if not getattr(robot, "has_active_viewer", False):
            for _ in range(50):
                if hasattr(robot, "step"):
                    robot.step()

    curr_pos, _ = robot.get_pose()
    curr_rpy = robot.get_imu()
    return (
        f"Motion completed. Current position: x={curr_pos[0]:.2f}m, y={curr_pos[1]:.2f}m, "
        f"heading={curr_rpy[2]:.1f}°"
    )


# ---------------------------------------------------------------------------
# Individual Simple Locomotion Tools (Easily reasoned by LLMs)
# ---------------------------------------------------------------------------

def walk_forward(distance_meters: float = 0.2) -> str:
    """
    Walk forward by a specified distance in meters.
    Use this to approach objects, advance along paths, or move toward a room.
    """
    return walk_relative(distance_meters=abs(distance_meters), turn_degrees=0.0)


def walk_backward(distance_meters: float = 0.2) -> str:
    """
    Walk backward by a specified distance in meters.
    Use this to back away from an obstacle or adjust positioning.
    """
    return walk_relative(distance_meters=-abs(distance_meters), turn_degrees=0.0)


def turn_left(degrees: float = 90.0) -> str:
    """
    Rotate body counter-clockwise (to the left) by a given number of degrees (default: 90° quarter turn).
    """
    return walk_relative(distance_meters=0.0, turn_degrees=-abs(degrees))


def turn_right(degrees: float = 90.0) -> str:
    """
    Rotate body clockwise (to the right) by a given number of degrees (default: 90° quarter turn).
    """
    return walk_relative(distance_meters=0.0, turn_degrees=abs(degrees))


# ---------------------------------------------------------------------------
# High-Level Navigation & Docking Tools (Non-Blocking & Interruptible)
# ---------------------------------------------------------------------------

from app.navigation import NavigationManager

_nav_manager: Optional[NavigationManager] = None


def get_nav_manager() -> NavigationManager:
    global _nav_manager
    if _nav_manager is None:
        _nav_manager = NavigationManager(get_robot)
    return _nav_manager


def navigate_to_landmark(landmark_name: str) -> str:
    """
    Begin non-blocking autonomous navigation to a domestic landmark.
    The robot will navigate in the background while reporting progress updates.
    Supported landmarks: 'front_door', 'charging_dock', 'living_room', 'kitchen'.
    """
    landmark_key = landmark_name.strip().lower().replace(" ", "_")
    if landmark_key not in config.landmarks:
        valid_keys = ", ".join(config.landmarks.keys())
        return f"Unknown landmark '{landmark_name}'. Valid landmarks are: {valid_keys}"

    target_x, target_y = config.landmarks[landmark_key]
    nav = get_nav_manager()
    res = nav.start_navigation(landmark_name=landmark_key, target_x=target_x, target_y=target_y)

    if res.get("status") == "already_at_destination":
        return f"Already at landmark '{landmark_name}'."

    return (
        f"Navigation started towards '{landmark_name}' at ({target_x}, {target_y}). "
        f"Distance: {res['initial_distance_m']}m. Estimated ETA: ~{res['estimated_eta_s']}s. "
        f"You will receive live progress updates as the robot walks."
    )


def stop_navigation() -> str:
    """
    Immediately halt the robot and cancel any active navigation or walking movement.
    Call this whenever the user says 'Stop!', 'Wait!', or gives an urgent command.
    """
    nav = get_nav_manager()
    res = nav.cancel_navigation()
    if res.get("status") == "idle":
        robot = get_robot()
        robot.stop()
        robot.stand()
        return "Robot is already stopped."
    return f"Emergency stop executed: Navigation to '{res.get('landmark')}' cancelled. Robot brought to a safe stand."


def return_to_charging_dock() -> str:
    """
    Autonomously navigates to the home charging dock and lowers onto the
    custom metal charging plates once arrived.
    """
    robot = get_robot()
    nav = get_nav_manager()

    def on_dock_reached():
        print("[CHARGING DOCK] Arrived at dock! Crouching onto metal contact plate...")
        robot.crouch()

    res = nav.start_navigation(
        landmark_name="charging_dock",
        target_x=0.0,
        target_y=0.0,
        on_complete_callback=on_dock_reached,
    )

    if res.get("status") == "already_at_destination":
        robot.crouch()
        return "Already at charging dock. Crouched onto metal contact plates."

    return (
        f"Docking sequence initiated. Walking towards charging dock at (0.0, 0.0). "
        f"Initial distance: {res['initial_distance_m']}m, ETA: ~{res['estimated_eta_s']}s. "
        f"The robot will automatically crouch onto charging contacts upon arrival."
    )


# ---------------------------------------------------------------------------
# Multimodal Vision, Communication, & Posture Tools
# ---------------------------------------------------------------------------

def capture_camera_image() -> Dict[str, Any]:
    """
    Captures an RGB image from the robot's front eye-level camera.
    Returns the visual frame for the agent to inspect its surroundings,
    verify who is at the door, or confirm task completion.
    """
    robot = get_robot()
    if hasattr(robot, "get_camera_image"):
        png_bytes = robot.get_camera_image()
        b64_str = base64.b64encode(png_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_str}"
        return {
            "status": "success",
            "description": "Captured front camera frame.",
            "image_data_url": data_url,
        }
    else:
        return {
            "status": "unavailable",
            "description": "Camera capture is not supported on this robot hardware interface.",
        }


def perform_handshake() -> str:
    """
    Performs the official Waveshare paw wave / handshake gesture.
    The robot sits back onto its rear legs, raises its right front paw,
    and shakes it to greet someone.
    """
    robot = get_robot()
    robot.handshake()
    # Step through simulation playback if in simulation
    if hasattr(robot, "motion_queue"):
        while len(robot.motion_queue) > 0:
            if hasattr(robot, "step"):
                robot.step()
            time.sleep(0.002)
    return "Handshake paw gesture completed."


def announce_message(message: str) -> str:
    """
    Announces a spoken message through the robot's speaker or notification system.
    Use this to greet visitors, inform residents, or alert people in the house.
    """
    print(f"\n[ROBOT SPEAKER ANNOUNCEMENT]: \"{message}\"\n")
    return f"Message announced: '{message}'"


def read_telemetry() -> Dict[str, Any]:
    """
    Reads the robot's current sensory telemetry, including 3D world position (x, y, z),
    Euler orientation (roll, pitch, yaw) in degrees, and movement status.
    """
    robot = get_robot()
    pos, quat = robot.get_pose()
    rpy = robot.get_imu()
    return {
        "position_xyz": [round(float(v), 3) for v in pos],
        "orientation_rpy": [round(float(v), 1) for v in rpy],
        "quaternion_wxyz": [round(float(v), 4) for v in quat],
        "is_moving": getattr(robot, "is_moving", False),
    }
