"""
Unified Action Manager for WAVEGO Pro.
Runs physical actions (locomotion, gestures, navigation) sequentially in a single
dedicated background worker thread, emitting real-time progress and completion
events to the gateway without blocking the cognitive agent.
"""

import base64
import math
import time
import threading
from typing import Optional, Tuple, Callable
import requests

from robot_interface import BaseRobot
from app.config import config
from app.vfh_planner import VFHPlanner


class PhysicalStallDetector:
    """
    Unified physical stall supervisor for WAVEGO Pro.
    Tracks ground translation (x, y) when robot is commanded to walk.
    Triggers a stall if the robot is walking (is_moving and not is_turning_in_place)
    but fails to translate min_displacement_m over timeout_s.
    """

    def __init__(self, timeout_s: float = 6.0, min_displacement_m: float = 0.04):
        self.timeout_s = timeout_s
        self.min_displacement_m = min_displacement_m
        self.last_check_time = time.time()
        self.last_check_pos: Optional[Tuple[float, float]] = None

    def update(self, robot: BaseRobot, curr_pos: Tuple[float, float]) -> Tuple[bool, float]:
        """
        Updates the stall supervisor with the latest robot pose.
        Returns:
            (is_stalled: bool, disp_since_check: float)
        """
        now = time.time()
        if self.last_check_pos is None:
            self.last_check_pos = curr_pos
            self.last_check_time = now
            return False, 0.0

        is_walking = getattr(robot, "is_moving", False) and not getattr(robot, "is_turning_in_place", False)

        if is_walking:
            disp = math.hypot(curr_pos[0] - self.last_check_pos[0], curr_pos[1] - self.last_check_pos[1])
            if disp >= self.min_displacement_m:
                self.last_check_time = now
                self.last_check_pos = curr_pos
                return False, disp
            elif (now - self.last_check_time) >= self.timeout_s:
                return True, disp
            return False, disp
        else:
            # Stopped or executing in-place pivot: reset translation window
            self.last_check_time = now
            self.last_check_pos = curr_pos
            return False, 0.0


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
        self._disambiguation_event = threading.Event()
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
        self._disambiguation_event.set()
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
        self._emit_gateway_event(
            "action_progress",
            {"action": action_name, "progress_pct": 100.0, "status": "cancelled"},
        )

        return {"status": "cancelled", "action": action_name}

    def submit_visual_disambiguation_scores(self, score_left: float, score_right: float):
        """
        Applies the agent's visual preference scores to the active VFH planner.
        This sets an exponential decaying bias towards the favored corridor.
        """
        if hasattr(self, "_vfh_planner") and self._vfh_planner is not None:
            self._vfh_planner.set_visual_preference(score_left, score_right)
            self._vfh_planner.last_selected_heading_deg = None
            print(f"\n[VFH VISUAL PREFERENCE APPLIED] Left={score_left:.2f}, Right={score_right:.2f} (bias={self._vfh_planner.visual_bias_val:.2f})")
            self._disambiguation_event.set()
            return {"status": "success", "score_left": score_left, "score_right": score_right}
        return {"status": "error", "message": "No active VFH planner"}

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

        # Initial UI progress popup
        self._emit_gateway_event(
            "action_progress",
            {
                "action": action_name,
                "progress_pct": 0.0,
                "distance_m": distance_meters,
                "turn_degrees": turn_degrees,
                "status": "started",
            },
        )

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

        # Initial UI progress popup
        self._emit_gateway_event(
            "action_progress",
            {"action": gesture_name, "progress_pct": 0.0, "status": "started"},
        )

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

        eta = round(initial_dist / 0.11, 1)
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
        if hasattr(self, "_vfh_planner") and self._vfh_planner is not None:
            self._vfh_planner.reset()

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
    # Worker Implementations & Motion Primitives
    # -------------------------------------------------------------------------

    def _turn_relative(
        self,
        turn_degrees: float,
        speed: float = 35.0,
        progress_callback: Optional[Callable[[float], None]] = None,
        max_duration_s: Optional[float] = None,
    ) -> float:
        """
        Reusable in-place relative rotation primitive using IMU yaw integration.
        Turns by turn_degrees (positive = clockwise/right, negative = counter-clockwise/left).
        Returns actual unwrapped degrees turned.
        """
        if abs(turn_degrees) < 0.5 or self._cancel_event.is_set():
            return 0.0

        robot = self.get_robot()
        start_rpy = robot.get_imu()
        prev_yaw = start_rpy[2]
        unwrapped_turn = 0.0
        target_yaw_diff = abs(turn_degrees)

        turn_dir = speed if turn_degrees > 0 else -speed
        robot.walk(0.0, turn_dir)

        if max_duration_s is None:
            max_duration_s = max(10.0, (target_yaw_diff / 10.0) + 4.0)

        t_start = time.time()
        while (time.time() - t_start) < max_duration_s and not self._cancel_event.is_set():
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

            if progress_callback:
                pct = min(99.0, round((abs(unwrapped_turn) / target_yaw_diff) * 100.0, 1))
                progress_callback(pct)

            if abs(unwrapped_turn) >= target_yaw_diff:
                break

        robot.stand()
        if not getattr(robot, "has_active_viewer", False):
            for _ in range(30):
                if hasattr(robot, "step"):
                    robot.step()

        return unwrapped_turn

    def _relative_motion_worker(
        self,
        action_name: str,
        distance_meters: float,
        turn_degrees: float,
    ):
        robot = self.get_robot()
        try:
            start_pos, _ = robot.get_pose()

            # 1. In-place rotation using shared _turn_relative primitive
            if abs(turn_degrees) > 1.0 and not self._cancel_event.is_set():
                last_emit = [time.time()]

                def on_turn_progress(pct: float):
                    self.progress_pct = pct
                    now = time.time()
                    if (now - last_emit[0]) >= 0.25:
                        last_emit[0] = now
                        self._emit_gateway_event(
                            "action_progress",
                            {
                                "action": action_name,
                                "progress_pct": self.progress_pct,
                                "turn_degrees": turn_degrees,
                            },
                        )

                self._turn_relative(
                    turn_degrees=turn_degrees,
                    speed=35.0,
                    progress_callback=on_turn_progress,
                )

            # 2. Linear motion
            if abs(distance_meters) > 0.01 and not self._cancel_event.is_set():
                speed = 100.0 if distance_meters > 0 else -100.0
                robot.walk(speed, 0.0)
                target_dist = abs(distance_meters)
                t_start = time.time()
                last_emit = t_start

                # Physical stall supervision for relative walking
                stall_detector = PhysicalStallDetector(timeout_s=6.0, min_displacement_m=0.03)
                stalled = False

                while (time.time() - t_start) < 25.0 and not self._cancel_event.is_set():
                    if hasattr(robot, "step") and not getattr(robot, "has_active_viewer", False):
                        robot.step()
                    time.sleep(0.002)
                    curr_pos, _ = robot.get_pose()
                    traveled = math.hypot(curr_pos[0] - start_pos[0], curr_pos[1] - start_pos[1])
                    self.progress_pct = min(99.0, round((traveled / target_dist) * 100.0, 1))

                    now = time.time()
                    if (now - last_emit) >= 0.25:
                        last_emit = now
                        self._emit_gateway_event(
                            "action_progress",
                            {
                                "action": action_name,
                                "progress_pct": self.progress_pct,
                                "distance_m": distance_meters,
                            },
                        )

                    if traveled >= target_dist:
                        break

                    # Check for physical stall
                    is_stalled, _ = stall_detector.update(robot, (curr_pos[0], curr_pos[1]))
                    if is_stalled:
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
                self._emit_gateway_event(
                    "action_progress",
                    {"action": action_name, "progress_pct": 100.0, "status": "completed"},
                )
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
                initial_q_len = max(1, len(robot.motion_queue))
                last_emit = time.time()
                while len(robot.motion_queue) > 0 and not self._cancel_event.is_set():
                    if hasattr(robot, "step"):
                        robot.step()
                    time.sleep(0.002)

                    remaining = len(robot.motion_queue)
                    self.progress_pct = min(99.0, round(((initial_q_len - remaining) / initial_q_len) * 100.0, 1))
                    now = time.time()
                    if (now - last_emit) >= 0.25:
                        last_emit = now
                        self._emit_gateway_event(
                            "action_progress",
                            {"action": gesture_name, "progress_pct": self.progress_pct},
                        )

            if not self._cancel_event.is_set():
                robot.stand()
                self.progress_pct = 100.0
                self._emit_gateway_event(
                    "action_progress",
                    {"action": gesture_name, "progress_pct": 100.0, "status": "completed"},
                )
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
        stall_detector = PhysicalStallDetector(timeout_s=6.0, min_displacement_m=0.04)

        # Reactive LiDAR obstacle avoidance & goal-progress-gated detour tracking
        is_detouring = False
        detour_stagnant_start_time = 0.0
        min_goal_dist_achieved = initial_dist
        detour_timeout = getattr(config, "lidar_detour_timeout_s", 15.0)
        min_goal_progress_delta = 0.04  # 4 cm progress towards landmark resets detour timeout
        obstacle_dist_thresh = getattr(config, "lidar_obstacle_distance_m", 0.45)
        emergency_stop_thresh = getattr(config, "lidar_emergency_stop_m", 0.16)

        # Visual disambiguation tracking & cooldown
        last_ambiguity_time = 0.0
        ambiguity_cooldown = getattr(config, "vfh_ambiguity_cooldown_s", 12.0)

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
                        "action_progress",
                        {
                            "action": "navigate_to_landmark",
                            "landmark": landmark_name,
                            "progress_pct": 100.0,
                            "status": "completed",
                        },
                    )

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

                # Closed-loop direct goal heading calculation
                target_heading_rad = math.atan2(dy, dx)
                target_heading_deg = math.degrees(target_heading_rad)

                # Query 360 DTOF LiDAR scan
                lidar_scan = None
                if hasattr(robot, "get_lidar_scan"):
                    lidar_scan = robot.get_lidar_scan(num_samples=72)

                # Initialize or reuse VFH Planner
                if not hasattr(self, "_vfh_planner") or self._vfh_planner is None:
                    self._vfh_planner = VFHPlanner(
                        num_bins=getattr(config, "lidar_num_samples", 72),
                        safe_distance_m=obstacle_dist_thresh,
                        emergency_stop_m=emergency_stop_thresh,
                        density_threshold_high=getattr(config, "vfh_density_threshold_high", 0.15),
                        density_threshold_low=getattr(config, "vfh_density_threshold_low", 0.06),
                        weight_goal=getattr(config, "vfh_weight_goal", 2.0),
                        weight_current_heading=getattr(config, "vfh_weight_current_heading", 0.5),
                        weight_prev_steering=getattr(config, "vfh_weight_prev_steering", 1.0),
                        robot_radius_m=0.18,
                        smoothing_tau_s=getattr(config, "vfh_smoothing_tau_s", 1.0),
                        flank_clearance_m=getattr(config, "vfh_flank_clearance_m", 0.10),
                        safety_margin_deg=getattr(config, "vfh_safety_margin_deg", 10.0),
                        ambiguity_threshold=getattr(config, "vfh_ambiguity_threshold", 0.05),
                        bifurcation_min_divergence_deg=getattr(config, "vfh_bifurcation_min_divergence_deg", 60.0),
                        visual_bias_weight=getattr(config, "vfh_visual_bias_weight", 2.5),
                        visual_bias_decay_s=getattr(config, "vfh_visual_bias_decay_s", 5.0),
                    )

                vfh_result = None
                if lidar_scan and "ranges" in lidar_scan:
                    vfh_result = self._vfh_planner.compute_steering(
                        lidar_ranges=lidar_scan["ranges"],
                        current_yaw_deg=curr_yaw,
                        target_heading_deg=target_heading_deg,
                        dist_to_goal=dist,
                        timestamp=now,
                    )

                if vfh_result:
                    is_detouring = vfh_result["is_detouring"]
                    heading_err = vfh_result["heading_error_deg"]
                    chosen_heading = vfh_result["chosen_heading_deg"]
                    min_clearance = vfh_result["min_distance_m"]
                    polar_density = vfh_result["polar_density"]

                    # Emergency stop condition
                    if vfh_result.get("emergency_stop", False):
                        print(f"\n[LIDAR EMERGENCY STOP] Obstacle at {min_clearance:.2f}m! Halting...")
                        robot.stop()
                        robot.stand()
                        self._emit_gateway_event(
                            "action_stalled",
                            {
                                "action": "navigate_to_landmark",
                                "landmark": landmark_name,
                                "distance_remaining_m": round(dist, 2),
                                "current_pos": [round(curr_x, 3), round(curr_y, 3)],
                                "reason": "lidar_emergency_stop_obstacle_too_close",
                                "min_clearance_m": min_clearance,
                                "sectors": lidar_scan.get("sectors", {}) if lidar_scan else {},
                            },
                        )
                        break

                    # Update action details for real-time telemetry streaming (and polar heatmap)
                    self._action_details["lidar"] = {
                        "min_distance_m": min_clearance,
                        "sectors": lidar_scan.get("sectors", {}),
                        "is_detouring": is_detouring,
                        "vfh_polar_density": polar_density,
                        "chosen_heading_deg": chosen_heading,
                        "heading_error_deg": heading_err,
                        "target_heading_deg": round(float(target_heading_deg), 1),
                        "score_left": vfh_result.get("score_left", 0.0),
                        "score_right": vfh_result.get("score_right", 0.0),
                        "is_ambiguous": vfh_result.get("is_ambiguous", False),
                        "active_visual_bias": vfh_result.get("active_visual_bias", 0.0),
                    }

                    # Bifurcation Ambiguity Routine:
                    # When obstacle head-on presents equally open left & right corridors,
                    # pause forward walk, pivot to left snapshot angle, take photo, pivot to right snapshot angle, take photo,
                    # return to heading, and notify cognitive agent to evaluate corridor traversability.
                    if (
                        is_detouring
                        and vfh_result.get("is_ambiguous", False)
                        and (now - last_ambiguity_time) >= ambiguity_cooldown
                        and not self._cancel_event.is_set()
                    ):
                        last_ambiguity_time = now
                        score_l = vfh_result.get("score_left", 0.0)
                        score_r = vfh_result.get("score_right", 0.0)
                        print(f"\n[VFH BIFURCATION AMBIGUITY] Left score={score_l:.2f}, Right score={score_r:.2f} (|diff| <= {config.vfh_ambiguity_threshold})! Initiating visual snapshot routine...")

                        robot.stop()
                        robot.stand()

                        # Instantaneous wide-camera view extraction (Option C):
                        # Front ultra-wide 160+ deg lens captures left and right flank views simultaneously
                        # without requiring 3 slow in-place mechanical body turns.
                        img_left_bytes, img_right_bytes = robot.get_disambiguation_views()

                        # Encode frames as base64 for gateway event payload
                        b64_left = base64.b64encode(img_left_bytes).decode("utf-8") if img_left_bytes else ""
                        b64_right = base64.b64encode(img_right_bytes).decode("utf-8") if img_right_bytes else ""

                        # Dispatch bifurcation disambiguation event to gateway
                        self._disambiguation_event.clear()
                        self._emit_gateway_event(
                            "bifurcation_disambiguation_required",
                            {
                                "landmark": landmark_name,
                                "current_pos": [round(curr_x, 3), round(curr_y, 3)],
                                "score_left": score_l,
                                "score_right": score_r,
                                "image_left_b64": b64_left,
                                "image_right_b64": b64_right,
                            },
                        )

                        # Stand in place and wait for cognitive agent to evaluate corridor preference (timeout: 12.0s)
                        print("[VFH BIFURCATION AMBIGUITY] Standing by for agent's corridor preference...")
                        wait_start = time.time()
                        max_wait_s = 12.0
                        while (
                            not self._disambiguation_event.is_set()
                            and not self._cancel_event.is_set()
                            and (time.time() - wait_start) < max_wait_s
                        ):
                            robot.stand()
                            if hasattr(robot, "step") and not getattr(robot, "has_active_viewer", False):
                                robot.step()
                            time.sleep(0.01)

                        if self._disambiguation_event.is_set():
                            print("[VFH BIFURCATION AMBIGUITY] Agent preference received! Resuming navigation with applied bias.")
                        else:
                            print("[VFH BIFURCATION AMBIGUITY] Wait timed out. Proceeding with default VFH steering.")
                else:
                    is_detouring = False
                    heading_err = (target_heading_deg - curr_yaw + 180.0) % 360.0 - 180.0

                # Detour Timeout Management
                # The clock ticks ONLY when an obstacle is within range AND no forward progress
                # toward the target landmark has occurred for detour_timeout seconds.
                # When a new min_goal_dist_achieved is achieved, the clock is RESET!
                if is_detouring:
                    if dist < (min_goal_dist_achieved - min_goal_progress_delta):
                        # Moving closer to goal landmark despite obstacle nearby! Reset timeout clock.
                        min_goal_dist_achieved = dist
                        detour_stagnant_start_time = now
                    else:
                        if detour_stagnant_start_time == 0.0:
                            detour_stagnant_start_time = now
                            print(f"\n[VFH OBSTACLE DETOUR] Obstacle ahead. Steering via VFH valley at {chosen_heading:.1f}° (dist_to_goal={dist:.2f}m)...")

                        detour_stagnant_elapsed = now - detour_stagnant_start_time
                        if detour_stagnant_elapsed >= detour_timeout:
                            print(f"\n[VFH DETOUR TIMEOUT] Stagnant near obstacle for {detour_stagnant_elapsed:.1f}s (> {detour_timeout:.1f}s) without goal progress! Stopping navigation to wake agent...")
                            robot.stop()
                            robot.stand()
                            self._emit_gateway_event(
                                "action_stalled",
                                {
                                    "action": "navigate_to_landmark",
                                    "landmark": landmark_name,
                                    "distance_remaining_m": round(dist, 2),
                                    "current_pos": [round(curr_x, 3), round(curr_y, 3)],
                                    "reason": "detour_timeout_obstacle_blocked",
                                    "detour_duration_s": round(detour_stagnant_elapsed, 1),
                                    "sectors": lidar_scan.get("sectors", {}) if lidar_scan else {},
                                },
                            )
                            break
                else:
                    if detour_stagnant_start_time != 0.0:
                        print(f"\n[VFH CLEAR] Path open! Resuming direct goal heading to '{landmark_name}'.")
                        detour_stagnant_start_time = 0.0
                    min_goal_dist_achieved = min(min_goal_dist_achieved, dist)

                # Physical Displacement Stall Detection
                # Uses unified PhysicalStallDetector to verify continuous ground translation
                is_stalled, disp_since_check = stall_detector.update(robot, (curr_x, curr_y))
                if is_stalled:
                    print(f"\n[PHYSICAL STALL DETECTED] Robot commanded to walk but stationary (disp={disp_since_check*100:.1f}cm in {stall_detector.timeout_s:.1f}s)! Stopping...")
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
                            "displacement_m": round(disp_since_check, 3),
                        },
                    )
                    break

                # Continuous VFH-Guided Quadruped Locomotion
                # Normalize heading error to [-180, 180]
                while heading_err > 180.0:
                    heading_err -= 360.0
                while heading_err < -180.0:
                    heading_err += 360.0

                # Dynamic Locomotion: Interspersed In-Place Turns + Pure Forward Strides
                # (No strafing or diagonal skidding; head always points toward path of motion)
                if not hasattr(self, "_is_aligning_heading"):
                    self._is_aligning_heading = False
                    self._align_start_time = 0.0

                # Hysteresis thresholds: initiate turn when error > 12 deg, exit turn when aligned within 6 deg
                align_entry_thresh = 12.0
                align_exit_thresh = 6.0
                min_turn_hold_s = 0.25  # Prevent rapid micro-toggling between turn and forward stride

                if not self._is_aligning_heading:
                    if abs(heading_err) > align_entry_thresh and dist > 0.20:
                        self._is_aligning_heading = True
                        self._align_start_time = time.time()
                else:
                    elapsed_turn = time.time() - self._align_start_time
                    if abs(heading_err) <= align_exit_thresh and elapsed_turn >= min_turn_hold_s:
                        self._is_aligning_heading = False

                if self._is_aligning_heading and dist > 0.20:
                    # Pure crisp in-place pivot to align head with target corridor
                    # In Waveshare kinematics: negative lr turns CCW (Left), positive turns CW (Right)
                    turn_spd = getattr(config, "nav_turn_speed", 25.0)
                    detour_turn_spd = getattr(config, "nav_detour_turn_speed", 20.0)
                    active_turn_speed = detour_turn_spd if is_detouring else turn_spd
                    turn_cmd = -active_turn_speed if heading_err > 0 else active_turn_speed
                    robot.walk(fb=0.0, lr=turn_cmd)
                else:
                    # Pure forward stride with 100% sagittal traction (zero lateral slip)
                    walk_spd = getattr(config, "nav_walk_speed", 85.0)
                    detour_spd = getattr(config, "nav_detour_speed", 70.0)
                    forward_speed = walk_spd if not is_detouring else detour_spd
                    robot.walk(fb=forward_speed, lr=0.0)

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
