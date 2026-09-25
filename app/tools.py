"""
ADK Tools for WAVEGO Pro Quadruped Robot.
Wraps BaseRobot hardware and simulation calls into typed LLM-callable functions.
"""

import math
import time
from typing import Any, Dict, Optional
from google.genai import types as genai_types
from google.adk.tools import ToolContext
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
        # Allocate enough time for full rotation (~12 deg/sec) plus margin
        max_turn_time = max(15.0, (target_yaw_diff / 10.0) + 5.0)
        t_start = time.time()

        # Continuous phase-unwrapped rotation tracking (handles 180°, 360°, and multiple spins)
        unwrapped_turn = 0.0
        prev_yaw = initial_yaw

        while (time.time() - t_start) < max_turn_time:
            if hasattr(robot, "step") and not getattr(robot, "has_active_viewer", False):
                robot.step()
            time.sleep(0.002)

            curr_yaw = robot.get_imu()[2]
            delta = curr_yaw - prev_yaw
            while delta > 180.0:
                delta -= 360.0
            while delta < -180.0:
                delta += 360.0

            unwrapped_turn += delta
            prev_yaw = curr_yaw

            if abs(unwrapped_turn) >= target_yaw_diff:
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
# Action Manager Reference
# ---------------------------------------------------------------------------

from app.action_manager import ActionManager

_action_manager: Optional[ActionManager] = None


def get_action_manager() -> ActionManager:
    global _action_manager
    if _action_manager is None:
        _action_manager = ActionManager(get_robot)
    return _action_manager


# ---------------------------------------------------------------------------
# Individual Locomotion Tools
# ---------------------------------------------------------------------------

def turn_left(degrees: float = 90.0, tool_context: Optional[ToolContext] = None) -> str:
    """Rotate body counter-clockwise (to the left) by a given number of degrees."""
    if tool_context:
        tool_context.actions.skip_summarization = True
    deg = abs(degrees)
    get_action_manager().start_relative_motion("turn_left", turn_degrees=-deg)
    return f"Initiated turn_left ({deg:.1f}°)."


def turn_right(degrees: float = 90.0, tool_context: Optional[ToolContext] = None) -> str:
    """Rotate body clockwise (to the right) by a given number of degrees."""
    if tool_context:
        tool_context.actions.skip_summarization = True
    deg = abs(degrees)
    get_action_manager().start_relative_motion("turn_right", turn_degrees=deg)
    return f"Initiated turn_right ({deg:.1f}°)."


# ---------------------------------------------------------------------------
# High-Level Navigation & Docking Tools
# ---------------------------------------------------------------------------

from app.environment_map import get_environment_map


def navigate_to_landmark(target_name: str, tool_context: Optional[ToolContext] = None) -> str:
    """Use this tool when there is a specific named landmark you want to go to such as 'living_room', 'kitchen', 'front_door', or 'charging_dock'."""
    if tool_context:
        tool_context.actions.skip_summarization = True

    landmark_key = target_name.strip().lower().replace(" ", "_")
    if landmark_key not in config.landmarks:
        valid_keys = ", ".join(config.landmarks.keys())
        return f"Unknown landmark '{target_name}'. Valid landmarks are: {valid_keys}"

    target_x, target_y = config.landmarks[landmark_key]
    mgr = get_action_manager()
    res = mgr.start_navigation(landmark_name=landmark_key, target_x=target_x, target_y=target_y)

    if res.get("status") == "already_at_destination":
        return f"Already at landmark '{target_name}'."

    return (
        f"Navigation started towards landmark '{target_name}' at ({target_x}, {target_y}). "
        f"Distance: {res['initial_dist_m'] if 'initial_dist_m' in res else res.get('distance_m', 0.0):.2f}m, ETA: ~{res.get('eta_s', res.get('eta_seconds', 0.0))}s."
    )


def navigate_to_coordinate(x: float, y: float, tool_context: Optional[ToolContext] = None) -> str:
    """Use this tool to travel to arbitrary coordinates x, y."""
    if tool_context:
        tool_context.actions.skip_summarization = True

    env_map = get_environment_map()
    is_valid, validation_msg = env_map.validate_target(x, y)
    if not is_valid:
        return f"Cannot navigate to ({x:.2f}, {y:.2f}): {validation_msg}"

    label = f"coordinate_{x:.2f}_{y:.2f}"
    mgr = get_action_manager()
    res = mgr.start_navigation(landmark_name=label, target_x=x, target_y=y)

    if res.get("status") == "already_at_destination":
        return f"Already at coordinate ({x:.2f}, {y:.2f})."

    dist = res.get("initial_dist_m", res.get("distance_m", 0.0))
    eta = res.get("eta_s", res.get("eta_seconds", 0.0))
    return f"Navigation started towards coordinate ({x:.2f}, {y:.2f}). Distance: {dist:.2f}m, ETA: ~{eta}s."


def stop_action(tool_context: Optional[ToolContext] = None) -> str:
    """Immediately halt the robot and cancel any active movement, navigation, or gesture."""
    # Do NOT set skip_summarization = True here: halting is an immediate, synchronous operation
    # and the agent may immediately issue a succeeding action (e.g. navigating to greet a guest).
    mgr = get_action_manager()
    res = mgr.cancel_action()
    if res.get("status") == "idle":
        return "Robot is already stopped and standing safely."

    action_name = res.get("action", "action")
    details = res.get("details", {})
    if action_name == "navigate_to_landmark" and "landmark" in details:
        desc = f"Navigation toward '{details['landmark']}'"
    elif action_name in ("walk_forward", "walk_backward") and "distance_m" in details:
        desc = f"{action_name} ({details['distance_m']:.2f}m)"
    elif action_name in ("turn_left", "turn_right") and "turn_degrees" in details:
        desc = f"{action_name} ({details['turn_degrees']:.1f}°)"
    else:
        desc = f"Action '{action_name}'"

    return f"Emergency stop executed: {desc} cancelled. Robot is standing safely."


def return_to_charging_dock(tool_context: Optional[ToolContext] = None) -> str:
    """Autonomously navigate to the home charging dock and lower onto charging contacts."""
    if tool_context:
        tool_context.actions.skip_summarization = True

    robot = get_robot()
    mgr = get_action_manager()

    def on_dock_reached():
        print("[CHARGING DOCK] Arrived at dock! Crouching onto metal contact plate...")
        robot.crouch()

    res = mgr.start_navigation(
        landmark_name="charging_dock",
        target_x=0.0,
        target_y=0.0,
        on_complete_callback=on_dock_reached,
    )

    if res.get("status") == "already_at_destination":
        robot.crouch()
        return "Already at charging dock. Crouched onto metal contact plates."

    return (
        f"Docking sequence initiated towards (0.0, 0.0). "
        f"Initial distance: {res['initial_distance_m']}m, ETA: ~{res['estimated_eta_s']}s."
    )


# ---------------------------------------------------------------------------
# Multimodal Vision, Communication, & Posture Tools
# ---------------------------------------------------------------------------

def capture_camera_image() -> Dict[str, Any]:
    """
    Captures an RGB image from the robot's front eye-level camera.
    Returns a multimodal image Part for Gemini vision inspection,
    allowing the agent to inspect its surroundings, identify visitors,
    or check for obstacles.
    """
    robot = get_robot()
    png_bytes = robot.get_camera_image()
    if png_bytes is not None:
        return {
            "status": "success",
            "description": "Captured front camera frame.",
            "image": genai_types.Part.from_bytes(
                data=png_bytes,
                mime_type="image/png",
            ),
        }
    else:
        return {
            "status": "unavailable",
            "description": "Camera capture is not supported or not connected on this robot hardware interface.",
        }


def perform_handshake(tool_context: Optional[ToolContext] = None) -> str:
    """Performs the official Waveshare paw wave / handshake gesture."""
    if tool_context:
        tool_context.actions.skip_summarization = True

    robot = get_robot()
    get_action_manager().start_gesture("handshake", robot.handshake)
    return "Initiated handshake paw gesture."


def announce_message(message: str) -> str:
    """
    Announces a spoken message through the robot's speaker or notification system.
    Use this to greet visitors, inform residents, or alert people in the house.
    """
    print(f"\n[ROBOT SPEAKER ANNOUNCEMENT]: \"{message}\"\n")
    return f"Message announced: '{message}'"


def read_telemetry() -> Dict[str, Any]:
    """
    Reads the robot's current sensory telemetry, including 2D floor position (x, y),
    Euler orientation (roll, pitch, yaw) in degrees, cardinal heading, and valid floor bounds.
    Yaw reference: 0° = East (+X forward), +90° = North (+Y left), -90° = South (-Y right), +/-180° = West (-X rear).
    """
    robot = get_robot()
    pos, quat = robot.get_pose()
    rpy = robot.get_imu()
    yaw_deg = round(float(rpy[2]), 1)

    # Determine cardinal heading string
    # Normalizing yaw to [0, 360)
    norm_yaw = (yaw_deg + 360.0) % 360.0
    cardinals = [
        "East (+X)", "North-East", "North (+Y)", "North-West",
        "West (-X)", "South-West", "South (-Y)", "South-East"
    ]
    card_idx = int(round(norm_yaw / 45.0)) % 8
    cardinal_name = cardinals[card_idx]

    env_map = get_environment_map()
    summary = env_map.get_summary()

    return {
        "position_xy": [round(float(pos[0]), 3), round(float(pos[1]), 3)],
        "heading_degrees": yaw_deg,
        "cardinal_direction": f"Facing {cardinal_name} (Yaw: {yaw_deg:+.1f}°)",
        "orientation_rpy": [round(float(v), 1) for v in rpy],
        "is_moving": getattr(robot, "is_moving", False),
        "floorplan_bounds": summary.get("navigable_range"),
        "known_landmarks": {k: list(v) for k, v in config.landmarks.items()},
    }


def read_lidar_scan() -> Dict[str, Any]:
    """
    Reads the current 360-degree DTOF LiDAR laser scan from the robot's top turret.
    Returns obstacle clearances (in meters) across 8 directional sectors
    (front, front_left, left, rear_left, rear, rear_right, right, front_right)
    and the overall minimum clearance distance around the robot.
    """
    robot = get_robot()
    if hasattr(robot, "get_lidar_scan"):
        scan = robot.get_lidar_scan()
        if scan:
            return {
                "status": "success",
                "min_distance_m": scan.get("min_distance_m"),
                "sectors": scan.get("sectors"),
            }
    return {
        "status": "unavailable",
        "description": "LiDAR sensor is not available or not connected.",
    }


def set_corridor_preference(
    score_left: float,
    score_right: float,
    tool_context: Optional[ToolContext] = None,
) -> str:
    """
    Sets the robot's directional preference between left and right corridors when circumventing an obstacle.
    Both score_left and score_right must be float values between 0.0 (strongly obstructed / undesirable)
    and 1.0 (completely open / highly desirable).
    This establishes an exponentially decaying bias in the robot's real-time Vector Field Histogram (VFH) planner.
    """
    if tool_context:
        tool_context.actions.skip_summarization = True

    score_left = max(0.0, min(1.0, float(score_left)))
    score_right = max(0.0, min(1.0, float(score_right)))

    mgr = get_action_manager()
    res = mgr.submit_visual_disambiguation_scores(score_left, score_right)
    if res.get("status") == "success":
        favored = "Left" if score_left > score_right else ("Right" if score_right > score_left else "Balanced")
        return (
            f"Successfully applied visual corridor preference: Left={score_left:.2f}, Right={score_right:.2f}. "
            f"Favored corridor: {favored}. VFH planner guidance updated."
        )
    return "No active navigation or VFH planner found to apply preference."

