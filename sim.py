"""
WAVEGO Pro MuJoCo Simulation & High-Level Controller Interface.
Integrates the authentic closed-loop 4-bar parallel linkage mechanism
with the official Waveshare BodyCtrl kinematics and JSON command interface.
"""

import math
import time
from typing import Dict, Any, Tuple
import mujoco
import mujoco.viewer
import numpy as np

from wavego_kinematics import (
    WavegoKinematics,
    STAND_HEIGHT,
    WALK_HEIGHT_MIN,
    WALK_HEIGHT_MAX,
)

# Command Types from official json_cmd.h
CMD_STAND_UP = 111
CMD_BASIC_MOVE = 112
CMD_BASIC_FUNC = 113
CMD_SINGLE_LEG_CTRL = 114


class WavegoSim:
    """
    Simulation wrapper providing the official WAVEGO Pro API
    running on top of the authentic closed-loop 4-bar linkage MuJoCo model.
    """

    def __init__(self, xml_path: str = "models/wavego/scene.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.kinematics = WavegoKinematics()

        # Motion & Gait State
        self.move_fb = 0.0      # Forward / Backward (-100 to 100)
        self.move_lr = 0.0      # Turn Left / Right (-100 to 100)
        self.walk_cycle = 0.0   # Gait cycle phase [0, 1)
        self.gait_speed = 1.5   # Gait frequency (cycles/second)
        self.stand_h = STAND_HEIGHT
        self.is_moving = False

        # Actuator mapping (12 serial-bus servo actuators)
        self.actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]

        # Initialize to standing pose
        stand_deg = self.kinematics.stand(self.stand_h)
        stand_rad = self.kinematics.get_joint_angles_rad(stand_deg)
        self.data.ctrl[:] = stand_rad

        # Settle physics for 200 steps
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

    def update_gait(self, dt: float):
        """
        Updates joint position setpoints using authentic Waveshare gait generation.
        Uses triangular gait with dynamic center-of-mass shift during walking,
        and kinematic standing pose when idle.
        """
        self.is_moving = abs(self.move_fb) > 2.0 or abs(self.move_lr) > 2.0

        if self.is_moving:
            # Advance gait cycle
            # Speed scales with move_fb magnitude
            speed_factor = max(abs(self.move_fb), abs(self.move_lr)) / 50.0
            speed_factor = min(max(speed_factor, 0.5), 2.0)
            self.walk_cycle = (self.walk_cycle + self.gait_speed * speed_factor * dt) % 1.0

            # Determine turning command (-1: left, 1: right, 0: straight)
            if self.move_lr > 15.0:
                turn_cmd = 1
            elif self.move_lr < -15.0:
                turn_cmd = -1
            else:
                turn_cmd = 0

            # Direction angle (0 = forward, 180 = backward)
            dir_angle = 0.0 if self.move_fb >= 0.0 else 180.0

            goal_deg = self.kinematics.triangular_gait(
                self.walk_cycle, direction_angle=dir_angle, turn_cmd=turn_cmd
            )
        else:
            self.walk_cycle = 0.0
            goal_deg = self.kinematics.stand(self.stand_h)

        goal_rad = self.kinematics.get_joint_angles_rad(goal_deg)
        self.data.ctrl[:] = goal_rad

    def step(self, dt: float = None):
        """Advances physics and kinematics by one timestep."""
        if dt is None:
            dt = self.model.opt.timestep
        self.update_gait(dt)
        mujoco.mj_step(self.model, self.data)

    def send_cmd(self, cmd_dict: Dict[str, Any]):
        """
        Official WAVEGO JSON Command Handler.
        Directly compatible with ESP32 firmware commands.
        """
        cmd_type = cmd_dict.get("T")

        if cmd_type == CMD_STAND_UP:
            self.move_fb = 0.0
            self.move_lr = 0.0
            self.stand_h = STAND_HEIGHT

        elif cmd_type == CMD_BASIC_MOVE:
            self.move_fb = float(cmd_dict.get("FB", 0))
            self.move_lr = float(cmd_dict.get("LR", 0))

        elif cmd_type == CMD_BASIC_FUNC:
            func = cmd_dict.get("func", 0)
            if func == 1:  # Stay low / crouch
                self.stand_h = 75.0
                self.move_fb = 0.0
                self.move_lr = 0.0
            elif func == 5:  # Return normal stand
                self.stand_h = STAND_HEIGHT
                self.move_fb = 0.0
                self.move_lr = 0.0

    # High-level convenience methods for GenAI task planners
    def stand(self):
        """Commands robot to stand in stable stance."""
        self.send_cmd({"T": CMD_STAND_UP})

    def walk(self, fb: float = 50.0, lr: float = 0.0):
        """Commands robot to walk (fb: forward/backward, lr: turn left/right)."""
        self.send_cmd({"T": CMD_BASIC_MOVE, "FB": fb, "LR": lr})

    def stop(self):
        """Commands robot to stop walking."""
        self.send_cmd({"T": CMD_BASIC_MOVE, "FB": 0, "LR": 0})

    def crouch(self):
        """Commands robot to crouch low."""
        self.send_cmd({"T": CMD_BASIC_FUNC, "func": 1})

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (x, y, z) position and orientation quaternion."""
        pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        return pos, quat

    def launch_viewer(self, target_fps: int = 30):
        """
        Launches an interactive 3D visualizer window with decoupled physics
        and rendering to keep CPU/GPU utilization low.
        """
        render_interval = 1.0 / target_fps
        sim_dt = self.model.opt.timestep
        last_render_time = time.time()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

            while viewer.is_running():
                now = time.time()
                self.step()

                if now - last_render_time >= render_interval:
                    viewer.sync()
                    last_render_time = now

                elapsed = time.time() - now
                sleep_time = sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
