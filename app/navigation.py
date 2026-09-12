"""
Asynchronous Navigation Manager for WAVEGO Pro.
Runs closed-loop waypoint navigation in a background thread using real physical
telemetry from MuJoCo (sim) or IMU/odometry (hardware), emitting real-time
progress % and ETA feedback without blocking the cognitive agent.
"""

import math
import time
import threading
from typing import Optional, Tuple, Callable
import requests

from robot_interface import BaseRobot
from app.config import config


class NavigationManager:
    """
    Manages non-blocking navigation goals, closed-loop telemetry tracking,
    and real-time progress/ETA reporting.
    """

    def __init__(self, get_robot_fn: Callable[[], BaseRobot]):
        self.get_robot = get_robot_fn
        self._thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._current_goal: Optional[str] = None
        self._is_navigating: bool = False

        # Live telemetry metrics
        self.target_name: str = ""
        self.target_coords: Tuple[float, float] = (0.0, 0.0)
        self.start_distance: float = 0.0
        self.remaining_distance: float = 0.0
        self.progress_pct: float = 0.0
        self.eta_seconds: float = 0.0

    @property
    def is_navigating(self) -> bool:
        return self._is_navigating

    def start_navigation(
        self,
        landmark_name: str,
        target_x: float,
        target_y: float,
        on_complete_callback: Optional[Callable[[], None]] = None,
    ) -> dict:
        """
        Starts non-blocking navigation toward a target coordinate.
        Immediately returns initial telemetry, distance, and estimated ETA.
        """
        # Cancel any previous navigation goal in-flight
        self.cancel_navigation()

        robot = self.get_robot()
        curr_pos, _ = robot.get_pose()
        dx = target_x - curr_pos[0]
        dy = target_y - curr_pos[1]
        initial_dist = math.hypot(dx, dy)

        if initial_dist < 0.05:
            return {
                "status": "already_at_destination",
                "landmark": landmark_name,
                "distance_m": round(initial_dist, 3),
                "eta_seconds": 0.0,
            }

        self.target_name = landmark_name
        self.target_coords = (target_x, target_y)
        self.start_distance = initial_dist
        self.remaining_distance = initial_dist
        self.progress_pct = 0.0

        # Estimated initial speed: ~0.15 m/s typical walking crawl
        estimated_speed = 0.15
        self.eta_seconds = round(initial_dist / estimated_speed, 1)

        self._cancel_event.clear()
        self._is_navigating = True
        self._thread = threading.Thread(
            target=self._navigation_worker,
            args=(landmark_name, target_x, target_y, on_complete_callback),
            daemon=True,
            name="NavWorkerThread",
        )
        self._thread.start()

        return {
            "status": "started",
            "landmark": landmark_name,
            "target_coords": (target_x, target_y),
            "initial_distance_m": round(initial_dist, 2),
            "estimated_eta_s": self.eta_seconds,
        }

    def cancel_navigation(self) -> dict:
        """Immediately stops the robot and cancels the background navigation goal."""
        if not self._is_navigating:
            return {"status": "idle", "message": "No active navigation goal to cancel."}

        self._cancel_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        self._is_navigating = False
        robot = self.get_robot()
        robot.stop()
        robot.stand()

        # Emit cancellation event to gateway
        self._emit_gateway_event(
            "nav_cancelled",
            {"landmark": self.target_name, "remaining_distance_m": round(self.remaining_distance, 2)},
        )

        return {"status": "cancelled", "landmark": self.target_name}

    def _navigation_worker(
        self,
        landmark_name: str,
        target_x: float,
        target_y: float,
        on_complete_callback: Optional[Callable[[], None]],
    ):
        robot = self.get_robot()
        last_feedback_time = time.time()
        feedback_interval = 20.0  # Send feedback event every 2 seconds
        last_pos_for_vel, _ = robot.get_pose()
        last_vel_time = time.time()
        measured_speed = 0.06  # Rolling average ground speed (m/s) calibrated to WAVEGO trot

        # Stall Detection: detect if robot is walking into a wall/obstacle
        last_progress_time = time.time()
        last_progress_dist = self.start_distance
        stall_timeout = 6.0  # 6.0s window allows robot to execute in-place turns without false stall
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
                self.remaining_distance = dist
                now = time.time()

                # Check arrival condition (within 8 cm of target landmark)
                if dist <= 0.08:
                    robot.stop()
                    robot.stand()
                    self._is_navigating = False
                    self.progress_pct = 100.0
                    self.eta_seconds = 0.0

                    if on_complete_callback:
                        on_complete_callback()

                    self._emit_gateway_event(
                        "nav_completed",
                        {
                            "landmark": landmark_name,
                            "final_pos": [round(curr_x, 3), round(curr_y, 3)],
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
                    # Stalled into a wall or obstacle while walking! Stop motors and alert the agent
                    print(f"\n[NAVIGATION STALL DETECTED] Robot stalled moving to '{landmark_name}'! Stopping motors...")
                    robot.stop()
                    robot.stand()
                    self._is_navigating = False
                    self._emit_gateway_event(
                        "nav_stalled",
                        {
                            "landmark": landmark_name,
                            "distance_remaining_m": round(dist, 2),
                            "current_pos": [round(curr_x, 3), round(curr_y, 3)],
                            "reason": "movement_deadlock_or_obstacle",
                        },
                    )
                    break

                # Closed-loop heading calculation
                target_yaw = math.degrees(math.atan2(dy, dx))
                heading_err = target_yaw - curr_yaw
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
                if self.start_distance > 0.0:
                    raw_pct = ((self.start_distance - dist) / self.start_distance) * 100.0
                    self.progress_pct = max(0.0, min(99.0, round(raw_pct, 1)))

                self.eta_seconds = max(0.5, round(dist / max(measured_speed, 0.05), 1))

                # Periodic progress feedback broadcast
                if (now - last_feedback_time) >= feedback_interval:
                    last_feedback_time = now
                    self._emit_gateway_event(
                        "nav_feedback",
                        {
                            "landmark": landmark_name,
                            "progress_pct": self.progress_pct,
                            "distance_remaining_m": round(dist, 2),
                            "current_speed_mps": round(measured_speed, 2),
                            "eta_seconds": self.eta_seconds,
                        },
                    )

                time.sleep(0.002)

        except Exception as e:
            print(f"[NAV WORKER ERROR]: {e}")
        finally:
            self._is_navigating = False

    def _emit_gateway_event(self, event_type: str, payload: dict):
        """Asynchronously dispatches internal navigation feedback to the FastAPI gateway."""
        try:
            requests.post(
                f"http://127.0.0.1:{config.port}/events",
                json={
                    "source": "navigation_system",
                    "event_type": event_type,
                    "payload": payload,
                },
                timeout=0.5,
            )
        except Exception:
            pass
