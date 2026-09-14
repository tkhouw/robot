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


config = AgentConfig()
