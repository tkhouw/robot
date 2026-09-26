"""
Scout Quadruped Robot Protocol Definitions and Command Constants.
Standard serial JSON command dictionary matching onboard ESP32 controller.
"""

import json
from typing import Dict, Any, Union

# Official Command Codes (Config.h)
CMD_UGV_CTRL = 1
CMD_PT_CTRL = 133

# Servo / Kinematics Calibration & Control
CMD_JOINT_MIDDLE = 101
CMD_RELEASE_TORQUE = 102
CMD_SINGLE_SERVO_CTRL = 103
CMD_GET_JOINTS_ZERO = 104
CMD_SET_JOINTS_ZERO = 105
CMD_GET_CURRENT_POS = 106
CMD_SET_CURRENT_POS_ZERO = 107
CMD_CTRL_JOINT_ANGLE = 108
CMD_CTRL_JOINT_RAD = 109

# High-Level Motion Commands
CMD_STAND_UP = 110          # {"T": 110}
CMD_BASIC_MOVE = 111        # {"T": 111, "FB": 0, "LR": 0}
CMD_BASIC_FUNC = 112        # {"T": 112, "func": 1..5}
CMD_SINGLE_LEG_CTRL = 113   # {"T": 113, "leg": 1, "x": 16, "y": 90, "z": 25}
CMD_STAND_UP_HEIGHT = 114   # {"T": 114, "h": 95}
CMD_SET_INTERPOLATION_PARAMS = 115
CMD_SET_GAIT_PARAMS = 116

# Basic Function sub-codes (CMD_BASIC_FUNC / T=112)
FUNC_STAY_LOW = 1
FUNC_HANDSHAKE = 2
FUNC_JUMP = 3
FUNC_STEADY_ON = 4
FUNC_STEADY_OFF = 5


def format_cmd(cmd_dict: Dict[str, Any]) -> str:
    """Serializes a command dictionary into an official newline-terminated JSON string."""
    return json.dumps(cmd_dict, separators=(',', ':')) + '\n'


def parse_cmd(raw_cmd: Union[str, bytes, Dict[str, Any]]) -> Dict[str, Any]:
    """Parses a raw command string, bytes, or dictionary into a normalized dictionary."""
    if isinstance(raw_cmd, dict):
        return raw_cmd
    if isinstance(raw_cmd, bytes):
        raw_cmd = raw_cmd.decode('utf-8')
    return json.loads(raw_cmd.strip())
