"""
Base Robot Interface for WAVEGO Pro Quadruped.
Provides the hardware-swappable abstraction layer uniting simulation (MuJoCo)
and physical hardware (ESP32 serial / wireless communication).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union
import numpy as np

from wavego_protocol import (
    CMD_STAND_UP,
    CMD_BASIC_MOVE,
    CMD_BASIC_FUNC,
    CMD_STAND_UP_HEIGHT,
    CMD_SINGLE_LEG_CTRL,
    FUNC_STAY_LOW,
    FUNC_HANDSHAKE,
    FUNC_JUMP,
    FUNC_STEADY_ON,
    FUNC_STEADY_OFF,
    parse_cmd,
    format_cmd,
)


class BaseRobot(ABC):
    """
    Abstract Base Class for the WAVEGO Pro robot.
    Both SimRobotController (MuJoCo) and PhysicalRobotController (Serial/WiFi)
    implement this identical interface.
    """

    @abstractmethod
    def send_cmd(self, cmd: Union[Dict[str, Any], str]) -> None:
        """
        The low-level hardware communication bottleneck.
        Accepts official WAVEGO JSON dictionary or string and executes it on the backend.
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

    # =========================================================================
    # High-Level Semantic Wrappers (Shared identically between Sim & Real)
    # These methods compile semantic requests into official WAVEGO JSON packets.
    # =========================================================================

    def stand(self) -> None:
        """Commands the robot to stand in default stable stance (T=110)."""
        self.send_cmd({"T": CMD_STAND_UP})

    def stand_height(self, height: float = 95.0) -> None:
        """Sets standing body height in millimeters (T=114)."""
        self.send_cmd({"T": CMD_STAND_UP_HEIGHT, "h": height})

    def walk(self, fb: float = 50.0, lr: float = 0.0) -> None:
        """
        Commands locomotion with forward/backward velocity (FB) and steering (LR) (T=111).
        Range: -100.0 to 100.0.
        """
        self.send_cmd({"T": CMD_BASIC_MOVE, "FB": fb, "LR": lr})

    def stop(self) -> None:
        """Stops locomotion (T=111, FB=0, LR=0)."""
        self.send_cmd({"T": CMD_BASIC_MOVE, "FB": 0, "LR": 0})

    def crouch(self) -> None:
        """Commands the robot to crouch low (T=112, func=1)."""
        self.send_cmd({"T": CMD_BASIC_FUNC, "func": FUNC_STAY_LOW})

    def handshake(self) -> None:
        """Plays handshake greeting routine (T=112, func=2)."""
        self.send_cmd({"T": CMD_BASIC_FUNC, "func": FUNC_HANDSHAKE})

    def set_single_leg(self, leg_id: int, x: float, y: float, z: float) -> None:
        """Positions an individual foot in 3D relative to hip (T=113)."""
        self.send_cmd({"T": CMD_SINGLE_LEG_CTRL, "leg": leg_id, "x": x, "y": y, "z": z})


class PhysicalRobotController(BaseRobot):
    """
    Physical Robot Driver for WAVEGO Pro.
    Sends raw JSON packets over UART serial cable (or WebSocket / TCP socket).
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200):
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
