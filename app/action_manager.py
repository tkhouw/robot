"""
Unified Action Manager for WAVEGO Pro.
Runs physical actions (locomotion, gestures, navigation) sequentially in a single
dedicated background worker thread, emitting real-time progress and completion
events to the gateway without blocking the cognitive agent.
"""

import math
import time
import threading
from typing import Optional, Tuple, Callable
import requests

from robot_interface import BaseRobot
from app.config import config


class ActionManager:
    """
    Unified manager for all non-blocking robot actions.
    Executes actions sequentially in a dedicated worker thread, emitting real-time
    feedback and completion events to the gateway without blocking the cognitive agent.
    """

    def __init__(self, get_robot_fn: Callable[[], BaseRobot]):
        self.get_robot = get_robot_fn
        self._thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._active_action: Optional[str] = None
        self._action_details: dict = {}
        self.progress_pct: float = 0.0

    @property
    def is_busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def active_action(self) -> Optional[str]:
        return self._active_action if self.is_busy else None

    @property
    def action_details(self) -> dict:
        return self._action_details if self.is_busy else {}

    def cancel_action(self) -> dict:
        """Immediately halts robot motors and terminates any active background action."""
        if not self.is_busy:
            robot = self.get_robot()
            robot.stop()
            robot.stand()
            return {"status": "idle", "message": "No active action to cancel."}

        action_name = self._active_action or "action"
        self._cancel_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        self._active_action = None
        self._action_details = {}
        self.progress_pct = 0.0

        robot = self.get_robot()
        robot.stop()
        robot.stand()

        # Emit cancellation event to gateway
        self._emit_gateway_event(
            "action_cancelled",
            {"action": action_name},
        )

        return {"status": "cancelled", "action": action_name}

    def start_relative_motion(
        self,
        action_name: str,
        distance_meters: float = 0.0,
        turn_degrees: float = 0.0,
    ) -> dict:
        """
        Starts non-blocking linear motion or turning in background thread.
        Returns immediately with action metadata.
        """
        self.cancel_action()

        self._cancel_event.clear()
        self._active_action = action_name
        self._action_details = {
            "type": "relative_motion",
            "distance_m": distance_meters,
            "turn_degrees": turn_degrees,
        }
        self.progress_pct = 0.0

        self._thread = threading.Thread(
            target=self._relative_motion_worker,
            args=(action_name, distance_meters, turn_degrees),
            daemon=True,
            name="ActionWorkerThread",
        )
        self._thread.start()
        return {"status": "started", "action": action_name}

    def start_gesture(self, gesture_name: str, gesture_fn: Callable[[], None]) -> dict:
        """
        Starts a non-blocking gesture sequence (e.g. handshake paw wave).
        """
        self.cancel_action()

        self._cancel_event.clear()
        self._active_action = gesture_name
        self._action_details = {"type": "gesture", "name": gesture_name}
        self.progress_pct = 0.0

        self._thread = threading.Thread(
            target=self._gesture_worker,
            args=(gesture_name, gesture_fn),
            daemon=True,
            name="ActionWorkerThread",
        )
        self._thread.start()
        return {"status": "started", "action": gesture_name}

    def start_navigation(
        self,
        landmark_name: str,
        target_x: float,
        target_y: float,
        on_complete_callback: Optional[Callable[[], None]] = None,
    ) -> dict:
        """
        Starts non-blocking closed-loop navigation toward landmark coordinates.
        """
        self.cancel_action()

        robot = self.get_robot()
        curr_pos, _ = robot.get_pose()
        dx = target_x - curr_pos[0]
        dy = target_y - curr_pos[1]
        initial_dist = math.hypot(dx, dy)

        if initial_dist < 0.08:
            if on_complete_callback:
                on_complete_callback()
            return {
                "status": "already_at_destination",
                "landmark": landmark_name,
                "distance_m": round(initial_dist, 3),
                "eta_seconds": 0.0,
            }

        eta = round(initial_dist / 0.06, 1)
        self._cancel_event.clear()
        self._active_action = "navigate_to_landmark"
        self._action_details = {
            "type": "navigation",
            "landmark": landmark_name,
            "target": [target_x, target_y],
            "initial_dist_m": initial_dist,
            "remaining_dist_m": initial_dist,
            "eta_s": eta,
        }
        self.progress_pct = 0.0

        self._thread = threading.Thread(
            target=self._navigation_worker,
            args=(landmark_name, target_x, target_y, on_complete_callback),
            daemon=True,
            name="ActionWorkerThread",
        )
        self._thread.start()

        return {
            "status": "started",
            "landmark": landmark_name,
            "target_coords": (target_x, target_y),
            "initial_distance_m": round(initial_dist, 2),
            "estimated_eta_s": eta,
        }

    # -------------------------------------------------------------------------
    # Worker Implementations
    # -------------------------------------------------------------------------

    def _relative_motion_worker(
        self,
        action_name: str,
        distance_meters: float,
        turn_degrees: float,
    ):
        robot = self.get_robot()
        try:
            start_pos, _ = robot.get_pose()
            start_rpy = robot.get_imu()
            initial_yaw = start_rpy[2]

            # 1. In-place rotation
            if abs(turn_degrees) > 1.0 and not self._cancel_event.is_set():
                turn_dir = 35.0 if turn_degrees > 0 else -35.0
                robot.walk(0.0, turn_dir)
                target_yaw_diff = abs(turn_degrees)
                max_turn_time = max(15.0, (target_yaw_diff / 10.0) + 5.0)
                t_start = time.time()
                unwrapped_turn = 0.0
                prev_yaw = initial_yaw

                while (time.time() - t_start) < max_turn_time and not self._cancel_event.is_set():
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
                    self.progress_pct = min(99.0, round((abs(unwrapped_turn) / target_yaw_diff) * 100.0, 1))

                    if abs(unwrapped_turn) >= target_yaw_diff:
                        break

                robot.stand()
                if not getattr(robot, "has_active_viewer", False):
                    for _ in range(50):
                        if hasattr(robot, "step"):
                            robot.step()

            # 2. Linear motion
            if abs(distance_meters) > 0.01 and not self._cancel_event.is_set():
                speed = 50.0 if distance_meters > 0 else -50.0
                robot.walk(speed, 0.0)
                target_dist = abs(distance_meters)
                t_start = time.time()

                # Stall detection for relative walking
                last_progress_time = time.time()
                last_traveled = 0.0
                stall_timeout = 5.0
                stalled = False

                while (time.time() - t_start) < 25.0 and not self._cancel_event.is_set():
                    if hasattr(robot, "step") and not getattr(robot, "has_active_viewer", False):
                        robot.step()
                    time.sleep(0.002)
                    curr_pos, _ = robot.get_pose()
                    traveled = math.hypot(curr_pos[0] - start_pos[0], curr_pos[1] - start_pos[1])
                    self.progress_pct = min(99.0, round((traveled / target_dist) * 100.0, 1))

                    if traveled >= target_dist:
                        break

                    now = time.time()
                    if traveled > (last_traveled + 0.02):
                        last_progress_time = now
                        last_traveled = traveled
                    elif (now - last_progress_time) >= stall_timeout:
                        stalled = True
                        break

                robot.stand()
                if not getattr(robot, "has_active_viewer", False):
                    for _ in range(50):
                        if hasattr(robot, "step"):
                            robot.step()

                if stalled and not self._cancel_event.is_set():
                    curr_pos, _ = robot.get_pose()
                    self._emit_gateway_event(
                        "action_stalled",
                        {
                            "action": action_name,
                            "reason": "physical_obstruction",
                            "current_pos": [round(float(curr_pos[0]), 3), round(float(curr_pos[1]), 3)],
                        },
                    )
                    return

            if not self._cancel_event.is_set():
                curr_pos, _ = robot.get_pose()
                curr_rpy = robot.get_imu()
                self.progress_pct = 100.0
                res_desc = f"x={curr_pos[0]:.2f}m, y={curr_pos[1]:.2f}m, heading={curr_rpy[2]:.1f}°"
                self._emit_gateway_event(
                    "action_completed",
                    {
                        "action": action_name,
                        "result": res_desc,
                        "position": [round(float(curr_pos[0]), 3), round(float(curr_pos[1]), 3)],
                    },
                )
        except Exception as e:
            print(f"[ACTION WORKER ERROR in '{action_name}']: {e}")
        finally:
            self._active_action = None

    def _gesture_worker(self, gesture_name: str, gesture_fn: Callable[[], None]):
        robot = self.get_robot()
        try:
            gesture_fn()
            # Step simulation frames smoothly while gesture motion queue plays out
            if hasattr(robot, "motion_queue"):
                while len(robot.motion_queue) > 0 and not self._cancel_event.is_set():
                    if hasattr(robot, "step"):
                        robot.step()
                    time.sleep(0.002)

            if not self._cancel_event.is_set():
                robot.stand()
                self.progress_pct = 100.0
                curr_pos, _ = robot.get_pose()
                self._emit_gateway_event(
                    "action_completed",
                    {
                        "action": gesture_name,
                        "result": f"{gesture_name} gesture routine completed.",
                        "position": [round(float(curr_pos[0]), 3), round(float(curr_pos[1]), 3)],
                    },
                )
        except Exception as e:
            print(f"[GESTURE WORKER ERROR in '{gesture_name}']: {e}")
        finally:
            self._active_action = None

    def _navigation_worker(
        self,
        landmark_name: str,
        target_x: float,
        target_y: float,
        on_complete_callback: Optional[Callable[[], None]],
    ):
        robot = self.get_robot()
        last_feedback_time = time.time()
        feedback_interval = 2.0  # Send feedback event every 2 seconds
        last_pos_for_vel, _ = robot.get_pose()
        last_vel_time = time.time()
        measured_speed = 0.06  # Rolling average ground speed (m/s) calibrated to WAVEGO trot

        initial_dist = self._action_details.get("initial_dist_m", 1.0)
        last_progress_time = time.time()
        last_progress_dist = initial_dist
        stall_timeout = 6.0
        min_progress_delta = 0.03  # 3 cm threshold

        try:
            while not self._cancel_event.is_set():
                curr_pos, _ = robot.get_pose()
                curr_rpy = robot.get_imu()
                curr_x, curr_y = curr_pos[0], curr_pos[1]
                curr_yaw = curr_rpy[2]

                dx = target_x - curr_x
                dy = target_y - curr_y
                dist = math.hypot(dx, dy)
                self._action_details["remaining_dist_m"] = round(dist, 3)
                now = time.time()

                # Check arrival condition (within 8 cm of target landmark)
                if dist <= 0.08:
                    robot.stop()
                    robot.stand()
                    self.progress_pct = 100.0
                    self._action_details["eta_s"] = 0.0

                    if on_complete_callback:
                        on_complete_callback()

                    self._emit_gateway_event(
                        "action_completed",
                        {
                            "action": "navigate_to_landmark",
                            "landmark": landmark_name,
                            "result": f"Arrived at landmark '{landmark_name}'",
                            "position": [round(curr_x, 3), round(curr_y, 3)],
                        },
                    )
                    break

                # Stall Check: Detect if robot is physically stuck against an obstacle
                # Never trigger stalls while robot is executing an in-place turn
                if getattr(robot, "is_turning_in_place", False):
                    last_progress_time = now
                    last_progress_dist = dist
                elif dist < (last_progress_dist - min_progress_delta):
                    last_progress_time = now
                    last_progress_dist = dist
                elif (now - last_progress_time) >= stall_timeout:
                    print(f"\n[STALL DETECTED] Robot stalled navigating to '{landmark_name}'! Stopping motors...")
                    robot.stop()
                    robot.stand()
                    self._emit_gateway_event(
                        "action_stalled",
                        {
                            "action": "navigate_to_landmark",
                            "landmark": landmark_name,
                            "distance_remaining_m": round(dist, 2),
                            "current_pos": [round(curr_x, 3), round(curr_y, 3)],
                            "reason": "movement_deadlock_or_obstacle",
                        },
                    )
                    break

                # Closed-loop heading calculation
                target_heading_rad = math.atan2(dy, dx)
                target_heading_deg = math.degrees(target_heading_rad)

                heading_err = target_heading_deg - curr_yaw
                while heading_err > 180.0:
                    heading_err -= 360.0
                while heading_err < -180.0:
                    heading_err += 360.0

                # Steering vs Straight Walking logic
                # Note: In Waveshare kinematics, move_lr < 0 turns Left (yaw increases)
                # and move_lr > 0 turns Right (yaw decreases).
                if abs(heading_err) > 30.0 and dist > 0.20:
                    # Large heading error: turn in place first
                    turn_cmd = -35.0 if heading_err > 0 else 35.0
                    robot.walk(fb=0.0, lr=turn_cmd)
                else:
                    # Forward with smooth proportional heading correction
                    lr_correction = max(-25.0, min(25.0, -heading_err * 1.5))
                    robot.walk(fb=50.0, lr=lr_correction)

                # Advance physics in simulation if available (and if not driven by active 3D viewer)
                if hasattr(robot, "step") and not getattr(robot, "has_active_viewer", False):
                    robot.step()

                # Calculate true rolling speed and ETA
                now = time.time()
                dt_vel = now - last_vel_time
                if dt_vel >= 0.5:
                    disp = math.hypot(curr_x - last_pos_for_vel[0], curr_y - last_pos_for_vel[1])
                    if disp > 0.001:
                        instant_speed = disp / dt_vel
                        measured_speed = 0.7 * measured_speed + 0.3 * instant_speed
                    last_pos_for_vel = curr_pos
                    last_vel_time = now

                # Progress percentage based on true physical displacement
                if initial_dist > 0.0:
                    raw_pct = ((initial_dist - dist) / initial_dist) * 100.0
                    self.progress_pct = max(0.0, min(99.0, round(raw_pct, 1)))

                eta_s = max(0.5, round(dist / max(measured_speed, 0.05), 1))
                self._action_details["eta_s"] = eta_s

                # Periodic progress feedback broadcast
                if (now - last_feedback_time) >= feedback_interval:
                    last_feedback_time = now
                    self._emit_gateway_event(
                        "action_progress",
                        {
                            "action": "navigate_to_landmark",
                            "landmark": landmark_name,
                            "progress_pct": self.progress_pct,
                            "distance_remaining_m": round(dist, 2),
                            "current_speed_mps": round(measured_speed, 2),
                            "eta_seconds": eta_s,
                        },
                    )

                time.sleep(0.002)

        except Exception as e:
            print(f"[ACTION WORKER ERROR in navigation]: {e}")
        finally:
            self._active_action = None

    def _emit_gateway_event(self, event_type: str, payload: dict):
        """Asynchronously dispatches internal action events to the FastAPI gateway."""
        try:
            requests.post(
                f"http://127.0.0.1:{config.port}/events",
                json={
                    "source": "action_system",
                    "event_type": event_type,
                    "payload": payload,
                },
                timeout=0.5,
            )
        except Exception:
            pass
