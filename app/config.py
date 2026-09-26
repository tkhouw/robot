"""
Configuration settings for the WAVEGO Pro Cognitive Agent.
"""

from typing import Dict, Tuple
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM Settings (must be provided via environment or .env file)
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Default to DeepSeek multimodal vision model
    model_name: str = "deepseek/deepseek-v4-flash-vision-exp"
    # Maximum LLM inference rounds allowed per stimulus turn (ADK native safeguard)
    agent_max_llm_calls: int = 5

    @field_validator("openrouter_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "OPENROUTER_API_KEY must be set in your environment or .env file and cannot be empty or None.  See .env.example."
            )
        return v.strip()

    # Gateway Settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Passive Motion Sensor Settings
    motion_detection_fps: float = 2.0
    motion_diff_threshold: float = 25.0  # Mean pixel difference to trigger motion alert

    # Known Domestic Landmarks (x, y coordinates in meters)
    landmarks: Dict[str, Tuple[float, float]] = {
        "charging_dock": (-0.6, 0.0),
        "front_door": (4.4, 0.0),
        "living_room": (1.1, 0.85),
        "kitchen": (3.5, -1.2),
    }

    # Locomotion & Speed Settings (Configurable limits based on physical 4-bar linkage dynamics)
    # gait_speed: Base gait frequency (cycles/second). Tested safe range: [1.0, 2.4].
    # Values > 2.4 exceed the 40ms bus servo bandwidth and cause foot slippage.
    gait_speed: float = 2.0

    # walk_range_mm: Stride reach in mm (forward/backward stroke). Tested safe range: [30.0, 55.0].
    # Values > 55.0 approach mechanical linkage singularities and cause pitch tilt.
    walk_range_mm: float = 30.0

    # walk_lift_mm: Vertical foot ground clearance in mm. Tested safe range: [8.0, 18.0].
    # Higher values provide cleaner obstacle/floor liftoff at high frequencies.
    walk_lift_mm: float = 13.0

    # nav_walk_speed: Forward FB speed for straight landmark navigation. Range: [20.0, 100.0].
    nav_walk_speed: float = 90.0

    # nav_detour_speed: Forward FB speed during obstacle avoidance detours. Range: [20.0, 90.0].
    nav_detour_speed: float = 90.0

    # nav_turn_speed: In-place turning LR speed for normal course corrections. Range: [15.0, 50.0].
    nav_turn_speed: float = 20.0

    # nav_detour_turn_speed: In-place turning LR speed during obstacle detours. Range: [10.0, 35.0].
    # Slower turning prevents high rotational inertia and tipping during sharp pivots.
    nav_detour_turn_speed: float = 20.0

    # sim_time_warp: Wall-clock simulation speed multiplier. Tested range: [0.5, 3.0].
    # 1.0 = true real-time, 1.5 = 1.5x real-time speed.
    sim_time_warp: float = 1.0

    # Navigation Goal Arrival Tolerances (meters)
    nav_goal_tolerance_m: float = 0.08  # Distance threshold for arrival at an unobstructed goal
    nav_obstructed_goal_tolerance_m: float = 0.35  # Proximity threshold to accept arrival if target point is physically blocked by an obstacle

    # 360 DTOF LiDAR & Obstacle Avoidance Settings
    lidar_num_samples: int = 72
    lidar_max_range_m: float = 12.0
    lidar_obstacle_distance_m: float = 0.60  # Distance threshold to initiate detour circumvention
    lidar_emergency_stop_m: float = 0.15     # Hard collision stop buffer
    lidar_detour_timeout_s: float = 120.0     # Time off-track navigating around obstacle before waking agent

    # Vector Field Histogram (VFH) Dynamic Filter & Safety Buffer Settings
    vfh_smoothing_tau_s: float = 1.0         # Exponential smoothing time constant (seconds) for temporal density filtering
    vfh_flank_clearance_m: float = 0.10      # Lateral flank safety buffer (distance to left/right hips)
    vfh_safety_margin_deg: float = 10.0      # Angular clearance margin (degrees) from obstacle boundaries in valleys
    vfh_density_threshold_high: float = 0.15 # Density threshold to enter detour mode
    vfh_density_threshold_low: float = 0.06  # Density threshold to exit detour mode / identify valleys

    # VFH Candidate Heading Selection Weights
    vfh_weight_goal: float = 2.0             # Weight favoring candidate valleys aligned with goal landmark
    vfh_weight_current_heading: float = 0.5  # Weight favoring current body orientation
    vfh_weight_prev_steering: float = 1.0    # Momentum/damping weight penalizing changes from previous steering direction (lower = more agile)

    # VFH Visual Disambiguation & Exponential Bias Decay
    vfh_ambiguity_threshold: float = 0.25    # Trigger threshold for |Left_Score - Right_Score| under which obstacle is ambiguous
    vfh_bifurcation_min_divergence_deg: float = 60.0 # Minimum angular divergence between left and right candidates to qualify as bifurcation
    vfh_snapshot_turn_angle_deg: float = 65.0 # In-place pivot angle in degrees for capturing left and right snapshots
    vfh_ambiguity_cooldown_s: float = 30.0   # Cooldown period (seconds) before another visual disambiguation can trigger
    vfh_visual_bias_weight: float = 15.5      # Cost function weight multiplier for agent's visual preference
    vfh_visual_bias_decay_s: float = 10.0     # Characteristic exponential decay timescale (seconds) for visual preference bias

    # Battery Simulation Settings
    battery_initial_pct: float = 100.0
    battery_drain_rate_moving: float = 0.35  # % per second walking (~4.7 min continuous walking)
    battery_drain_rate_idle: float = 0.02    # % per second standing (~83 min idle standby)
    battery_charge_rate: float = 10.0        # % per second charging on dock (10 seconds to full charge)
    battery_low_threshold_pct: float = 20.0  # Threshold triggering P1 low-battery alert
    battery_critical_threshold_pct: float = 5.0 # Critical threshold triggering urgent P1 alert

config = AgentConfig()
