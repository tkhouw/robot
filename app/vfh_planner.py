"""
Vector Field Histogram (VFH) Local Obstacle Avoidance Planner.
Implements smoothed polar obstacle density (POD) mapping, candidate valley identification,
and momentum-damped target heading selection to eliminate oscillations.
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class VFHPlanner:
    """
    True Vector Field Histogram (VFH) local obstacle avoidance planner.
    """

    def __init__(
        self,
        num_bins: int = 72,
        safe_distance_m: float = 0.55,
        emergency_stop_m: float = 0.16,
        density_threshold_high: float = 0.15,
        density_threshold_low: float = 0.08,
        weight_goal: float = 2.0,
        weight_current_heading: float = 0.5,
        weight_prev_steering: float = 3.0,
        robot_radius_m: float = 0.18,
        smoothing_tau_s: float = 1.0,
        flank_clearance_m: float = 0.16,
        safety_margin_deg: float = 25.0,
    ):
        self.num_bins = num_bins
        self.bin_width_deg = 360.0 / num_bins
        self.safe_distance_m = safe_distance_m
        self.emergency_stop_m = emergency_stop_m
        self.tau_high = density_threshold_high
        self.tau_low = density_threshold_low
        self.robot_radius_m = robot_radius_m
        self.smoothing_tau_s = max(0.01, smoothing_tau_s)
        self.flank_clearance_m = flank_clearance_m
        self.safety_margin_deg = safety_margin_deg

        # Cost function weights (anti-oscillation tuning)
        self.weight_goal = weight_goal
        self.weight_current_heading = weight_current_heading
        self.weight_prev_steering = weight_prev_steering  # Damps steering chatter

        # State tracking
        self.last_selected_heading_deg: Optional[float] = None
        self.is_detouring: bool = False
        self.detour_start_time: float = 0.0

        # Temporal Exponential Smoothing State (low-pass filter across time)
        self.temporal_density: Optional[np.ndarray] = None
        self.last_update_time: Optional[float] = None

        # Circular Gaussian spatial smoothing kernel
        self.kernel = np.array([0.06, 0.24, 0.40, 0.24, 0.06], dtype=np.float64)

    def reset(self):
        """Resets planner state between navigation runs."""
        self.last_selected_heading_deg = None
        self.is_detouring = False
        self.detour_start_time = 0.0
        self.temporal_density = None
        self.last_update_time = None

    def compute_steering(
        self,
        lidar_ranges: List[float],
        current_yaw_deg: float,
        target_heading_deg: float,
        dist_to_goal: float,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Computes the optimal steering heading using True VFH.

        Args:
            lidar_ranges: Array of distance measurements (num_bins elements, swept -180 to +180 deg).
            current_yaw_deg: Robot body yaw in degrees (-180 to +180).
            target_heading_deg: Direct heading toward goal landmark in degrees (-180 to +180).
            dist_to_goal: Euclidean distance to goal in meters.
            timestamp: Current timestamp (seconds) for exponential temporal smoothing.

        Returns:
            Dictionary containing:
                - 'chosen_heading_deg': Recommended world heading setpoint.
                - 'heading_error_deg': Steering error relative to current body yaw.
                - 'is_detouring': True if circumventing an obstacle.
                - 'emergency_stop': True if an obstacle is within emergency stop margin.
                - 'polar_density': Smoothed obstacle density array (for UI heatmap).
                - 'min_distance_m': Closest obstacle clearance.
        """
        import time

        curr_time = timestamp if timestamp is not None else time.time()
        n = self.num_bins
        if len(lidar_ranges) != n:
            # Resample or pad if sample count differs
            ranges_arr = np.interp(
                np.linspace(0, len(lidar_ranges) - 1, n),
                np.arange(len(lidar_ranges)),
                lidar_ranges,
            )
        else:
            ranges_arr = np.array(lidar_ranges, dtype=np.float64)

        min_dist = float(np.min(ranges_arr)) if len(ranges_arr) > 0 else self.safe_distance_m

        # Emergency collision stop check
        if min_dist <= self.emergency_stop_m and dist_to_goal > 0.15:
            return {
                "chosen_heading_deg": current_yaw_deg,
                "heading_error_deg": 0.0,
                "is_detouring": True,
                "emergency_stop": True,
                "polar_density": [1.0] * n,
                "min_distance_m": min_dist,
            }

        # 1. Calculate raw obstacle density per bin with physical robot footprint enlargement
        # Each obstacle at distance d is enlarged by angular margin gamma = arcsin(r_robot / d)
        # This prevents corners from scraping against the shoulders or rear legs
        raw_density = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d = ranges_arr[i]
            if d < self.safe_distance_m and d > 0.01:
                certainty = ((self.safe_distance_m - d) / self.safe_distance_m) ** 2
                raw_density[i] = max(raw_density[i], certainty)

                # Angular enlargement for physical body clearance
                sin_gamma = min(1.0, self.robot_radius_m / max(d, self.robot_radius_m))
                gamma_deg = math.degrees(math.asin(sin_gamma))
                enlarge_bins = int(math.ceil(gamma_deg / self.bin_width_deg))

                for offset in range(1, enlarge_bins + 1):
                    left_idx = (i - offset) % n
                    right_idx = (i + offset) % n
                    # Taper density across enlarged radius
                    enlarge_certainty = certainty * (1.0 - 0.3 * (offset / (enlarge_bins + 1)))
                    raw_density[left_idx] = max(raw_density[left_idx], enlarge_certainty)
                    raw_density[right_idx] = max(raw_density[right_idx], enlarge_certainty)

        # 2. Circular periodic convolution for Gaussian spatial smoothing
        k_len = len(self.kernel)
        pad = k_len // 2
        padded = np.concatenate([raw_density[-pad:], raw_density, raw_density[:pad]])
        instant_spatial_density = np.convolve(padded, self.kernel, mode="valid")

        # 3. Temporal Exponential Smoothing across time: D_t = (1 - alpha) * D_{t-1} + alpha * D_instant
        # alpha = 1 - exp(-dt / tau). As dt -> 0, alpha -> dt / tau.
        # This prevents single-frame noise/chatter from causing twitchy steering maneuvers.
        if self.temporal_density is None or self.last_update_time is None:
            self.temporal_density = instant_spatial_density.copy()
        else:
            dt = max(0.001, min(1.0, curr_time - self.last_update_time))
            alpha = float(1.0 - math.exp(-dt / self.smoothing_tau_s))
            self.temporal_density = (1.0 - alpha) * self.temporal_density + alpha * instant_spatial_density

        self.last_update_time = curr_time
        smoothed_density = self.temporal_density

        # 4. Bin angles in world frame
        # Lidar rays sweep from -180 to +180 relative to body
        rel_angles_deg = np.linspace(-180.0, 180.0, n, endpoint=False)
        world_bin_angles_deg = (current_yaw_deg + rel_angles_deg + 180.0) % 360.0 - 180.0

        # Direct goal angle error
        goal_diff = (target_heading_deg - current_yaw_deg + 180.0) % 360.0 - 180.0
        # Find index corresponding closest to goal angle in robot frame
        goal_bin_idx = int(np.argmin(np.abs((rel_angles_deg - goal_diff + 180.0) % 360.0 - 180.0)))

        # 5. Check if the direct line to goal is unobstructed (Dual-Threshold Hysteresis)
        # Direct path check covers bins within +/- 35 degrees of goal direction (wider corridor)
        half_window = max(3, int(35.0 / self.bin_width_deg))
        direct_blocked = False

        # Threshold depends on whether we are already in detour mode:
        # - Entry threshold: tau_high (0.15)
        # - Exit threshold: tau_low (0.06)
        threshold = self.tau_low if self.is_detouring else self.tau_high

        for offset in range(-half_window, half_window + 1):
            idx = (goal_bin_idx + offset) % n
            if smoothed_density[idx] > threshold:
                direct_blocked = True
                break

        # If already detouring, also require closest obstacle clearance before exiting detour:
        if self.is_detouring and not direct_blocked:
            if min_dist < self.safe_distance_m:
                direct_blocked = True

        # Unified Flank Clearance Guard (Body-Frame):
        # The robot's physical flanks are located at relative angles:
        # Left Flank: [+40°, +120°], Right Flank: [-120°, -40°]
        # If an obstacle encroaches within flank_clearance_m of either flank, turning toward that side
        # would sweep the robot's rear legs/hips into the corner.
        flank_min_clearance_m = self.flank_clearance_m
        left_flank_min = float("inf")
        right_flank_min = float("inf")

        for i in range(n):
            rel_ang = rel_angles_deg[i]
            d = ranges_arr[i]
            if 40.0 <= rel_ang <= 120.0:
                left_flank_min = min(left_flank_min, d)
            elif -120.0 <= rel_ang <= -40.0:
                right_flank_min = min(right_flank_min, d)

        # If direct path is otherwise clear, check if turning toward the goal would scrape an encroaching flank:
        if not direct_blocked:
            if goal_diff > 5.0 and left_flank_min < flank_min_clearance_m:
                direct_blocked = True  # Turning left would swing right rear into obstacle
            elif goal_diff < -5.0 and right_flank_min < flank_min_clearance_m:
                direct_blocked = True  # Turning right would swing left rear into obstacle

        if not direct_blocked:
            # Direct path is open! Head directly to goal.
            self.is_detouring = False
            self.last_selected_heading_deg = target_heading_deg
            err = (target_heading_deg - current_yaw_deg + 180.0) % 360.0 - 180.0
            return {
                "chosen_heading_deg": round(float(target_heading_deg), 1),
                "heading_error_deg": round(float(err), 1),
                "is_detouring": False,
                "emergency_stop": False,
                "polar_density": [round(float(v), 3) for v in smoothed_density],
                "min_distance_m": round(min_dist, 2),
            }

        # 6. Direct path is blocked -> Enter Detour Mode & extract candidate valleys
        self.is_detouring = True

        # Group contiguous clear bins (density <= tau_low) into discrete open valleys
        # A bin is clear if smoothed_density <= tau_low
        is_clear = (smoothed_density <= self.tau_low)

        valleys = []
        in_valley = False
        v_start = 0

        # Handle circular wrap-around by finding transitions
        for i in range(n):
            if is_clear[i] and not in_valley:
                in_valley = True
                v_start = i
            elif not is_clear[i] and in_valley:
                in_valley = False
                valleys.append((v_start, i - 1))

        if in_valley:
            # Check if wraps around index 0
            if valleys and valleys[0][0] == 0:
                first_start, first_end = valleys.pop(0)
                valleys.append((v_start, first_end + n))
            else:
                valleys.append((v_start, n - 1))

        # Generate candidate steering directions from each open valley
        # Authentic VFH algorithm:
        # - Wide valley: candidates are border + s_max (safety clearance)
        # - Narrow valley: candidate is valley center
        s_max_bins = max(2, int(self.safety_margin_deg / self.bin_width_deg))  # Configurable obstacle clearance margin
        candidate_headings_world = []

        for v_start, v_end in valleys:
            v_width = v_end - v_start + 1
            if v_width >= (2 * s_max_bins):
                # Wide valley: safe directions near both boundaries + goal if inside
                c1_idx = (v_start + s_max_bins) % n
                c2_idx = (v_end - s_max_bins) % n
                c1_deg = world_bin_angles_deg[c1_idx]
                c2_deg = world_bin_angles_deg[c2_idx]

                # Check deflection relative to current body (ignore rear > 135 deg)
                if abs(rel_angles_deg[c1_idx]) <= 135.0:
                    candidate_headings_world.append(c1_deg)
                if abs(rel_angles_deg[c2_idx]) <= 135.0:
                    candidate_headings_world.append(c2_deg)

                # If goal direction falls inside this wide valley safely, add goal heading
                goal_in_valley = False
                diff_to_start = (goal_bin_idx - v_start) % n
                diff_to_end = (v_end - goal_bin_idx) % n
                if (diff_to_start + diff_to_end) == (v_width - 1):
                    if diff_to_start >= s_max_bins and diff_to_end >= s_max_bins:
                        candidate_headings_world.append(target_heading_deg)
            else:
                # Narrow valley: steer through the center
                mid_idx = (v_start + v_width // 2) % n
                if abs(rel_angles_deg[mid_idx]) <= 135.0:
                    candidate_headings_world.append(world_bin_angles_deg[mid_idx])

        # Fallback if no valid valley: pick the bin of lowest density
        if not candidate_headings_world:
            min_idx = int(np.argmin(smoothed_density))
            candidate_headings_world = [world_bin_angles_deg[min_idx]]

        # 6. Cost function evaluation across candidate valley directions
        best_heading = candidate_headings_world[0]
        best_cost = float("inf")
        prev_heading = self.last_selected_heading_deg if self.last_selected_heading_deg is not None else current_yaw_deg

        for cand_deg in candidate_headings_world:
            # Angular difference to goal
            diff_goal = abs((cand_deg - target_heading_deg + 180.0) % 360.0 - 180.0)
            # Angular difference to current robot heading
            diff_curr = abs((cand_deg - current_yaw_deg + 180.0) % 360.0 - 180.0)
            # Angular difference to previously chosen heading (MOMENTUM / HYSTERESIS DAMPING)
            diff_prev = abs((cand_deg - prev_heading + 180.0) % 360.0 - 180.0)

            cost = (
                self.weight_goal * diff_goal
                + self.weight_current_heading * diff_curr
                + self.weight_prev_steering * diff_prev
            )

            if cost < best_cost:
                best_cost = cost
                best_heading = cand_deg

        self.last_selected_heading_deg = best_heading
        heading_err = (best_heading - current_yaw_deg + 180.0) % 360.0 - 180.0

        return {
            "chosen_heading_deg": round(float(best_heading), 1),
            "heading_error_deg": round(float(heading_err), 1),
            "is_detouring": True,
            "emergency_stop": False,
            "polar_density": [round(float(v), 3) for v in smoothed_density],
            "min_distance_m": round(min_dist, 2),
        }
