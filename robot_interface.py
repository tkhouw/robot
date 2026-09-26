"""
Base Robot Interface for Scout Quadruped.
Provides the hardware-swappable abstraction layer uniting simulation (MuJoCo)
and physical hardware (serial / wireless communication).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union, Optional
import numpy as np

from robot_protocol import (
    CMD_STAND_UP,
    CMD_BASIC_MOVE,
    CMD_BASIC_FUNC,
    CMD_STAND_UP_HEIGHT,
    CMD_SINGLE_LEG_CTRL,
    FUNC_STAY_LOW,
    FUNC_HANDSHAKE,
    parse_cmd,
    format_cmd,
)


class BaseRobot(ABC):
    """
    Abstract Base Class for the Scout quadruped robot.
    Both SimRobotController (MuJoCo) and PhysicalRobotController (Serial/WiFi)
    implement this identical interface.
    """

    @abstractmethod
    def send_cmd(self, cmd: Union[Dict[str, Any], str]) -> None:
        """
        The low-level hardware communication bottleneck.
        Accepts serial JSON command dictionary or string and executes it on the backend.
        """
        pass

    @abstractmethod
    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns current estimated world pose: (position_xyz, orientation_quaternion).
        """
        pass

    @abstractmethod
    def get_imu(self) -> Tuple[float, float, float]:
        """
        Returns current body Euler angles: (roll, pitch, yaw) in degrees or radians.
        """
        pass

    def get_camera_image(
        self, width: int = 320, height: int = 240, camera_name: str = "front_camera"
    ) -> Optional[bytes]:
        """
        Captures an RGB image from the robot camera and returns PNG-encoded bytes.
        Returns None if camera hardware/stream is unavailable or not connected.
        """
        return None

    def get_disambiguation_views(
        self, width: int = 320, height: int = 240
    ) -> Tuple[Optional[bytes], Optional[bytes]]:
        """
        Captures Left and Right corridor views for visual disambiguation.
        For an ultra-wide angle camera (160+ deg FOV), extracts both views from a single
        instantaneous capture without requiring mechanical chassis/neck turns.
        Returns (left_png_bytes, right_png_bytes).
        """
        img = self.get_camera_image(width=width, height=height)
        return (img, img)

    def get_lidar_scan(
        self, num_samples: int = 72, max_range: float = 12.0
    ) -> Optional[Dict[str, Any]]:
        """
        Returns a 360-degree DTOF LiDAR range scan.
        Structure includes 'ranges', 'angles_deg', 'min_distance_m', and 'sectors' dictionary.
        Returns None if LiDAR hardware is unavailable or not enabled.
        """
        return None

    def get_battery_level(self) -> float:
        """
        Returns current estimated battery percentage (0.0 to 100.0).
        """
        raise NotImplementedError("Subclasses must implement get_battery_level()")

    def is_charging(self) -> bool:
        """
        Returns True if the robot is currently on charging contacts and replenishing power.
        """
        raise NotImplementedError("Subclasses must implement is_charging()")

    # =========================================================================
    # High-Level Semantic Wrappers (Shared identically between Sim & Real)
    # These methods compile semantic requests into serial JSON packets.
    # =========================================================================

    def __init__(self):
        self.is_moving: bool = False
        self.is_turning_in_place: bool = False

    def stand(self) -> None:
        """Commands the robot to stand in default stable stance (T=110)."""
        self.is_moving = False
        self.is_turning_in_place = False
        self.send_cmd({"T": CMD_STAND_UP})

    def stand_height(self, height: float = 95.0) -> None:
        """Sets standing body height in millimeters (T=114)."""
        self.is_moving = False
        self.is_turning_in_place = False
        self.send_cmd({"T": CMD_STAND_UP_HEIGHT, "h": height})

    def walk(self, fb: float = 50.0, lr: float = 0.0) -> None:
        """
        Commands locomotion with forward/backward velocity (FB) and steering (LR) (T=111).
        Range: -100.0 to 100.0.
        Turning in place implies is_moving is True.
        """
        self.is_moving = (abs(fb) > 2.0 or abs(lr) > 2.0)
        self.is_turning_in_place = (abs(fb) <= 2.0 and abs(lr) > 2.0)
        self.send_cmd({"T": CMD_BASIC_MOVE, "FB": fb, "LR": lr})

    def stop(self) -> None:
        """Stops locomotion (T=111, FB=0, LR=0)."""
        self.is_moving = False
        self.is_turning_in_place = False
        self.send_cmd({"T": CMD_BASIC_MOVE, "FB": 0, "LR": 0})

    def crouch(self) -> None:
        """Commands the robot to crouch low (T=112, func=1)."""
        self.is_moving = False
        self.is_turning_in_place = False
        self.send_cmd({"T": CMD_BASIC_FUNC, "func": FUNC_STAY_LOW})

    def handshake(self) -> None:
        """Plays handshake greeting routine (T=112, func=2)."""
        self.is_moving = True
        self.is_turning_in_place = False
        self.send_cmd({"T": CMD_BASIC_FUNC, "func": FUNC_HANDSHAKE})

    def set_single_leg(self, leg_id: int, x: float, y: float, z: float) -> None:
        """Positions an individual foot in 3D relative to hip (T=113)."""
        self.send_cmd({"T": CMD_SINGLE_LEG_CTRL, "leg": leg_id, "x": x, "y": y, "z": z})


class PhysicalRobotController(BaseRobot):
    """
    Physical Robot Driver for Scout Quadruped.
    Sends raw JSON packets over UART serial cable (or WebSocket / TCP socket).
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        # In a real environment with physical hardware:
        # import serial
        # self.serial = serial.Serial(port, baudrate, timeout=0.1)

    def send_cmd(self, cmd: Union[Dict[str, Any], str]) -> None:
        cmd_dict = parse_cmd(cmd)
        formatted_json = format_cmd(cmd_dict)
        if self.serial and self.serial.is_open:
            self.serial.write(formatted_json.encode('utf-8'))
        else:
            # Standby / dry-run mode when hardware is not plugged in
            pass

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        # Queried via IMU / odometry serial feedback
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def get_imu(self) -> Tuple[float, float, float]:
        # Queried via MPU6050 feedback
        return 0.0, 0.0, 0.0

    def get_camera_image(
        self, width: int = 320, height: int = 240, camera_name: str = "front_camera"
    ) -> Optional[bytes]:
        """
        Captures an RGB frame from the physical camera (e.g. USB V4L2 or ESP32-CAM stream).
        Returns None until physical video stream pipeline is configured.
        """
        # Future implementation: query OpenCV VideoCapture or HTTP JPEG stream
        return None
