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
        "charging_dock": (0.0, 0.0),
        "front_door": (4.4, 0.0),
        "living_room": (1.0, 1.2),
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

    # 360 DTOF LiDAR & Obstacle Avoidance Settings
    lidar_num_samples: int = 72
    lidar_max_range_m: float = 12.0
    lidar_obstacle_distance_m: float = 0.60  # Distance threshold to initiate detour circumvention
    lidar_emergency_stop_m: float = 0.10     # Hard collision stop buffer
    lidar_detour_timeout_s: float = 15.0     # Time off-track navigating around obstacle before waking agent


config = AgentConfig()
