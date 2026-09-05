"""
WAVEGO Pro MuJoCo Simulation & High-Level Controller Interface
Ported to match the official Waveshare JSON Command Interface and BodyCtrl.cpp.
"""

import math
import time
import mujoco
import mujoco.viewer
import numpy as np

# Official WAVEGO Leg Linkage dimensions (in meters)
# Scaled from BodyCtrl.cpp
LINKAGE_A = 0.0400  # Thigh link length (m)
LINKAGE_CD = 0.0716 # Combined calf/shank length (m)

# Stance parameters from BodyCtrl.cpp (meters)
STAND_HEIGHT = 0.095    # Height of chassis above ground (m)
STEP_LIFT = 0.015       # Foot lift height during swing (m)
EXTENDED_X = 0.025      # Fore-aft stance offset (m)
EXTENDED_Y = 0.025      # Lateral stance offset (m)

# Command Types from official json_cmd.h
CMD_STAND_UP = 111
CMD_BASIC_MOVE = 112
CMD_BASIC_FUNC = 113
CMD_SINGLE_LEG_CTRL = 114


class WavegoSim:
    """
    Simulation wrapper providing the official WAVEGO Pro API
    running on top of a scaled MuJoCo quadruped model.
    """

    def __init__(self, xml_path="models/wavego/scene.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Reset to default standing keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Motion state
        self.move_fb = 0.0      # Forward / Backward (-100 to 100)
        self.move_lr = 0.0      # Turn Left / Right (-100 to 100)
        self.walk_cycle = 0.0   # Gait cycle phase [0, 1)
        self.gait_speed = 1.0   # Cycles per second
        self.stand_h = STAND_HEIGHT

        # Actuator names (12 joints)
        self.actuator_names = [
            "FL_hip", "FL_thigh", "FL_calf",
            "FR_hip", "FR_thigh", "FR_calf",
            "RL_hip", "RL_thigh", "RL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
        ]
        self.actuator_ids = [self.model.actuator(name).id for name in self.actuator_names]

    def single_leg_ik(self, x, y, z, leg_idx):
        """
        Inverse Kinematics for a 3-DOF leg.
        x: forward/backward, y: lateral (outward positive), z: downward distance
        leg_idx: 0=FL, 1=FR, 2=RL, 3=RR
        """
        # Abduction / hip roll angle
        # Positive angle swings outward
        hip_sign = 1.0 if leg_idx in [0, 2] else -1.0
        hip = math.atan2(y * hip_sign, z)

        effective_z = math.sqrt(y**2 + z**2)
        r = math.sqrt(x**2 + effective_z**2)
        l1 = LINKAGE_A
        l2 = LINKAGE_CD

        # Clamp reach to physical kinematics bounds
        r = min(max(r, 0.03), (l1 + l2) * 0.98)

        # Calf / knee angle (law of cosines)
        cos_knee = (l1**2 + l2**2 - r**2) / (2 * l1 * l2)
        cos_knee = min(max(cos_knee, -1.0), 1.0)
        knee_angle = -(math.pi - math.acos(cos_knee))

        # Thigh pitch angle
        alpha = math.atan2(x, effective_z)
        cos_thigh = (l1**2 + r**2 - l2**2) / (2 * l1 * r)
        cos_thigh = min(max(cos_thigh, -1.0), 1.0)
        thigh_angle = alpha + math.acos(cos_thigh)

        return hip, thigh_angle, knee_angle

    def update_gait(self, dt):
        """Generates stable 4-beat crawl/trot gait."""
        is_moving = abs(self.move_fb) > 1.0 or abs(self.move_lr) > 1.0

        if is_moving:
            self.walk_cycle = (self.walk_cycle + self.gait_speed * dt) % 1.0
        else:
            self.walk_cycle = 0.0

        # Leg phase offsets (crawl gait: each leg lifts in quarter-cycle turn)
        # 0: FL, 1: FR, 2: RL, 3: RR
        leg_phases = [
            (self.walk_cycle + 0.00) % 1.0,  # FL
            (self.walk_cycle + 0.50) % 1.0,  # FR
            (self.walk_cycle + 0.75) % 1.0,  # RL
            (self.walk_cycle + 0.25) % 1.0,  # RR
        ]

        stride = (self.move_fb / 100.0) * 0.035
        turn_stride = (self.move_lr / 100.0) * 0.020

        # Base nominal foot positions with wide stance
        base_offsets = [
            ( EXTENDED_X, EXTENDED_Y),   # FL
            ( EXTENDED_X, -EXTENDED_Y),  # FR
            (-EXTENDED_X, EXTENDED_Y),   # RL
            (-EXTENDED_X, -EXTENDED_Y),  # RR
        ]

        leg_targets = []
        for i, phase in enumerate(leg_phases):
            base_x, base_y = base_offsets[i]
            turn_dir = 1.0 if i in [0, 2] else -1.0

            # Foot trajectory
            if is_moving and phase < 0.25:
                # Swing phase: foot lifts and reaches forward
                swing_prog = phase / 0.25
                x = base_x + (stride + turn_stride * turn_dir) * (swing_prog - 0.5)
                z = self.stand_h - STEP_LIFT * math.sin(math.pi * swing_prog)
            else:
                # Stance phase: foot pushes ground backward
                stance_prog = (phase - 0.25) / 0.75 if is_moving else 0.5
                x = base_x - (stride + turn_stride * turn_dir) * (stance_prog - 0.5)
                z = self.stand_h

            y = abs(base_y)
            hip, thigh, calf = self.single_leg_ik(x, y, z, i)
            leg_targets.extend([hip, thigh, calf])

        for act_id, target in zip(self.actuator_ids, leg_targets):
            self.data.ctrl[act_id] = target

    def step(self, dt=None):
        """Advances physics by one timestep."""
        if dt is None:
            dt = self.model.opt.timestep
        self.update_gait(dt)
        mujoco.mj_step(self.model, self.data)

    def send_cmd(self, cmd_dict):
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
            if func == 1:  # Stay low
                self.stand_h = 0.065
                self.move_fb = 0.0
                self.move_lr = 0.0
            elif func == 5:  # Return normal stand
                self.stand_h = STAND_HEIGHT

    # High-level convenience methods for GenAI task planners
    def stand(self):
        """Commands robot to stand in stable stance."""
        self.send_cmd({"T": CMD_STAND_UP})

    def walk(self, fb=50, lr=0):
        """Commands robot to walk (fb: forward/backward, lr: turn)."""
        self.send_cmd({"T": CMD_BASIC_MOVE, "FB": fb, "LR": lr})

    def stop(self):
        """Commands robot to stop walking."""
        self.send_cmd({"T": CMD_BASIC_MOVE, "FB": 0, "LR": 0})

    def crouch(self):
        """Commands robot to crouch low."""
        self.send_cmd({"T": CMD_BASIC_FUNC, "func": 1})

    def get_pose(self):
        """Returns (x, y, z) position and orientation quaternion."""
        pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        return pos, quat

    def launch_viewer(self, target_fps=30):
        """
        Launches an interactive 3D visualizer window with decoupled physics
        and rendering to keep CPU/GPU utilization low.
        """
        render_interval = 1.0 / target_fps
        sim_dt = self.model.opt.timestep
        last_render_time = time.time()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Turn off shadows in viewer flags for extra performance
            viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

            while viewer.is_running():
                now = time.time()
                # Run physics step
                self.step()

                # Only render and sync display at target_fps (default 30 FPS)
                if now - last_render_time >= render_interval:
                    viewer.sync()
                    last_render_time = now

                # Sleep to maintain wall-clock real-time speed
                elapsed = time.time() - now
                sleep_time = sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
