"""
Passive Vision Motion Sensor for WAVEGO Pro Quadruped.
Runs an efficient local frame-differencing loop to detect visual changes
and fire events without consuming LLM API calls.
"""

import io
import time
import threading
from typing import Callable, Optional
import numpy as np
from PIL import Image

from robot_interface import BaseRobot
from app.config import config


class PassiveMotionDetector:
    """
    Samples camera frames at low FPS and calculates the mean absolute difference
    between consecutive grayscale frames. When movement exceeds threshold,
    triggers an event callback.
    """

    def __init__(
        self,
        robot: BaseRobot,
        on_motion_callback: Optional[Callable[[float], None]] = None,
        diff_threshold: float = None,
        fps: float = None,
    ):
        self.robot = robot
        self.on_motion_callback = on_motion_callback
        self.diff_threshold = diff_threshold or config.motion_diff_threshold
        self.fps = fps or config.motion_detection_fps

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._last_alert_time: float = 0.0
        self._cooldown_seconds: float = 5.0  # Prevent flooding events

    def start(self):
        """Starts the background frame differencing loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="MotionSensorThread")
        self._thread.start()

    def stop(self):
        """Stops the background detector."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _loop(self):
        sample_interval = 1.0 / self.fps

        while self._running:
            start_t = time.time()

            # Skip checking if the robot is currently walking/turning to prevent self-motion false positives
            if getattr(self.robot, "is_moving", False):
                self._prev_frame = None
                time.sleep(sample_interval)
                continue

            if hasattr(self.robot, "get_camera_image"):
                try:
                    img_bytes = self.robot.get_camera_image(width=160, height=120)
                    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # Grayscale
                    curr_frame = np.array(img, dtype=np.float32)

                    if self._prev_frame is not None:
                        # Mean pixel delta
                        diff = float(np.mean(np.abs(curr_frame - self._prev_frame)))

                        now = time.time()
                        if diff >= self.diff_threshold and (now - self._last_alert_time) > self._cooldown_seconds:
                            self._last_alert_time = now
                            if self.on_motion_callback:
                                self.on_motion_callback(diff)

                    self._prev_frame = curr_frame
                except Exception as e:
                    pass

            elapsed = time.time() - start_t
            sleep_time = sample_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
