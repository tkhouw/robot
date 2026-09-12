"""
WAVEGO Pro MuJoCo Simulation & High-Level Controller Interface.
Integrates the authentic closed-loop 4-bar parallel linkage mechanism
with the official Waveshare BodyCtrl kinematics and JSON command interface.
"""

import math
import time
import threading
from typing import Dict, Any, Tuple, Union, List
import mujoco
import mujoco.viewer
import numpy as np

from wavego_kinematics import (
    WavegoKinematics,
    STAND_HEIGHT,
    WALK_HEIGHT_MIN,
    WALK_HEIGHT_MAX,
)

from robot_interface import BaseRobot
from wavego_protocol import (
    CMD_STAND_UP,
    CMD_BASIC_MOVE,
    CMD_BASIC_FUNC,
    CMD_SINGLE_LEG_CTRL,
    CMD_STAND_UP_HEIGHT,
    CMD_CTRL_JOINT_ANGLE,
    CMD_CTRL_JOINT_RAD,
    CMD_SET_GAIT_PARAMS,
    FUNC_STAY_LOW,
    FUNC_HANDSHAKE,
    FUNC_STEADY_OFF,
    parse_cmd,
)


class WavegoSim(BaseRobot):
    """
    MuJoCo Simulation driver providing the official WAVEGO Pro API
    running on top of the authentic closed-loop 4-bar linkage model.
    Implements BaseRobot for complete hardware-swappability.
    """

    def __init__(self, xml_path: str = "models/wavego/scene.xml"):
        super().__init__()
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
        self.is_turning_in_place = False

        # Actuator mapping (12 serial-bus servo actuators)
        self.actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]

        # Initialize directly to precomputed standing keyframe
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.data.ctrl[:] = self.model.key_ctrl[0]
        else:
            stand_rad = self.kinematics.stand(self.stand_h)
            self.data.ctrl[:] = stand_rad

        # Initialize target and filtered control buffers
        self.target_ctrl = np.array(self.data.ctrl, dtype=np.float64).copy()
        self.filtered_ctrl = np.array(self.data.ctrl, dtype=np.float64).copy()
        self.servo_tau = 0.04  # 40ms physical servo mechanical time constant

        # Motion queue for sequential gestures (e.g. handshake)
        self.motion_queue: List[List[float]] = []

        # Thread safety lock & persistent renderer cache to prevent GLFW context recreation crashes
        self._lock = threading.Lock()
        self._renderers: Dict[tuple, mujoco.Renderer] = {}

        # Settle physics for 50 steps
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

    def update_gait(self, dt: float):
        """
        Updates joint position setpoints using authentic Waveshare gait generation.
        Uses simpleGait (diagonal trot) during walking, active motion queue for gestures,
        and target_ctrl when idle, with physical servo rate smoothing to prevent discontinuous kicks.
        """
        if self.motion_queue:
            self.walk_cycle = 0.0
            self.is_moving = False
            self.target_ctrl[:] = self.motion_queue.pop(0)
            target_rad = self.target_ctrl
        else:
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

                # Use authentic Waveshare default gait (GAIT_TYPE = 0: simpleGait)
                target_rad = np.array(
                    self.kinematics.simple_gait(
                        self.walk_cycle, direction_angle=dir_angle, turn_cmd=turn_cmd
                    )
                )
            else:
                self.walk_cycle = 0.0
                target_rad = self.target_ctrl

        # Smooth servo command interpolation (matches physical bus servo velocity profile)
        alpha = dt / (dt + self.servo_tau)
        self.filtered_ctrl += alpha * (target_rad - self.filtered_ctrl)
        self.data.ctrl[:] = self.filtered_ctrl

    def _get_renderer(self, width: int, height: int) -> mujoco.Renderer:
        """Retrieves or creates a cached persistent Renderer instance for the given resolution."""
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderers[key]

    def step(self, dt: float = None):
        """Advances physics and kinematics by one timestep."""
        if dt is None:
            dt = self.model.opt.timestep
        with self._lock:
            self.update_gait(dt)
            mujoco.mj_step(self.model, self.data)

    def send_cmd(self, cmd: Union[Dict[str, Any], str]):
        """
        Official WAVEGO JSON Command Handler.
        Directly compatible with ESP32 firmware commands.
        Accepts dict or raw JSON string and immediately executes kinematics.
        """
        cmd_dict = parse_cmd(cmd)
        cmd_type = cmd_dict.get("T")

        if cmd_type == CMD_STAND_UP:
            self.is_turning_in_place = False
            self.move_fb = 0.0
            self.move_lr = 0.0
            self.stand_h = STAND_HEIGHT
            # Immediately invoke WavegoKinematics to calculate 12-servo standing pose
            self.target_ctrl[:] = self.kinematics.stand(self.stand_h, rad=True)

        elif cmd_type == CMD_STAND_UP_HEIGHT:
            self.is_turning_in_place = False
            self.stand_h = float(cmd_dict.get("h", STAND_HEIGHT))
            self.move_fb = 0.0
            self.move_lr = 0.0
            # Immediately invoke WavegoKinematics to calculate variable height pose
            self.target_ctrl[:] = self.kinematics.stand(self.stand_h, rad=True)

        elif cmd_type == CMD_SINGLE_LEG_CTRL:
            # Single foot 3D positioning via closed-form IK: {"T": 113, "leg": 1..4, "x": ..., "y": ..., "z": ...}
            leg_id = int(cmd_dict.get("leg", 1))
            x = float(cmd_dict.get("x", 0.0))
            y = float(cmd_dict.get("y", STAND_HEIGHT))
            z = float(cmd_dict.get("z", 0.0))
            self.kinematics.single_leg_ctrl(leg_id, x, y, z)
            self.target_ctrl[:] = self.kinematics.get_joint_angles_rad()

        elif cmd_type == CMD_CTRL_JOINT_RAD:
            # Direct single joint radian command: {"T": 109, "joint": 0..11, "rad": ...}
            joint_idx = int(cmd_dict.get("joint", 0))
            rad_val = float(cmd_dict.get("rad", 0.0))
            if 0 <= joint_idx < 12:
                self.target_ctrl[joint_idx] = rad_val

        elif cmd_type == CMD_CTRL_JOINT_ANGLE:
            # Direct single joint angle (degrees) command: {"T": 108, "joint": 0..11, "angle": ...}
            joint_idx = int(cmd_dict.get("joint", 0))
            angle_val = float(cmd_dict.get("angle", 0.0))
            if 0 <= joint_idx < 12:
                self.kinematics.goal_angles[joint_idx] = angle_val
                self.target_ctrl[:] = self.kinematics.get_joint_angles_rad()

        elif cmd_type == CMD_SET_GAIT_PARAMS:
            # Set gait parameters: {"T": 116, ...}
            if "height" in cmd_dict:
                self.kinematics.walk_height = float(cmd_dict["height"])
            if "lift" in cmd_dict:
                self.kinematics.walk_lift = float(cmd_dict["lift"])
            if "range" in cmd_dict:
                self.kinematics.walk_range = float(cmd_dict["range"])
            if "extendedX" in cmd_dict:
                self.kinematics.walk_extended_x = float(cmd_dict["extendedX"])
            if "extendedZ" in cmd_dict:
                self.kinematics.walk_extended_z = float(cmd_dict["extendedZ"])

        elif cmd_type == CMD_BASIC_MOVE:
            self.move_fb = float(cmd_dict.get("FB", 0))
            self.move_lr = float(cmd_dict.get("LR", 0))
            self.is_turning_in_place = (abs(self.move_fb) <= 2.0 and abs(self.move_lr) > 2.0)

        elif cmd_type == CMD_BASIC_FUNC:
            self.is_turning_in_place = False
            func = cmd_dict.get("func", 0)
            if func == FUNC_STAY_LOW:  # Stay low / crouch
                self.motion_queue.clear()
                self.stand_h = 75.0
                self.move_fb = 0.0
                self.move_lr = 0.0
                self.target_ctrl[:] = self.kinematics.stand(self.stand_h, rad=True)
            elif func == FUNC_HANDSHAKE:  # Handshake gesture
                self.move_fb = 0.0
                self.move_lr = 0.0
                # Generate and queue the authentic Waveshare handshake sequence
                traj = self.kinematics.generate_handshake_trajectory()
                # Upsample/stretch trajectory frames slightly so the motion is smooth at 500 Hz
                # Each waypoint is held for ~5 physics steps (10ms)
                stretched_traj = []
                for wp in traj:
                    for _ in range(5):
                        stretched_traj.append(wp)
                self.motion_queue = stretched_traj
            elif func == FUNC_STEADY_OFF:  # Return normal stand
                self.motion_queue.clear()
                self.stand_h = STAND_HEIGHT
                self.move_fb = 0.0
                self.move_lr = 0.0
                self.target_ctrl[:] = self.kinematics.stand(self.stand_h, rad=True)

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (x, y, z) position and orientation quaternion [w, x, y, z]."""
        with self._lock:
            pos = self.data.qpos[0:3].copy()
            quat = self.data.qpos[3:7].copy()
        return pos, quat

    def get_imu(self) -> Tuple[float, float, float]:
        """
        Returns estimated Euler angles (roll, pitch, yaw) in degrees from quaternion.
        Matches onboard physical MPU6050 output.
        """
        with self._lock:
            w, x, y, z = self.data.qpos[3:7]
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(np.clip(sinp, -1.0, 1.0))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    def get_camera_image(
        self, width: int = 320, height: int = 240, camera_name: str = "front_camera"
    ) -> bytes:
        """
        Renders an RGB image from the specified camera ('front_camera' or 'tracking_cam')
        and returns PNG encoded bytes.
        """
        import io
        from PIL import Image

        with self._lock:
            renderer = self._get_renderer(width, height)
            renderer.update_scene(self.data, camera=camera_name)
            rgb_array = renderer.render()

        img = Image.fromarray(rgb_array)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def get_jpeg_frame(
        self, camera_name: str = "tracking_cam", width: int = 480, height: int = 360, quality: int = 75
    ) -> bytes:
        """
        Fast JPEG rendering for live web streaming over MJPEG.
        """
        import io
        from PIL import Image

        with self._lock:
            renderer = self._get_renderer(width, height)
            renderer.update_scene(self.data, camera=camera_name)
            rgb_array = renderer.render()

        img = Image.fromarray(rgb_array)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def launch_viewer(self, target_fps: int = 30, key_callback=None):
        """
        Launches an interactive 3D visualizer window with decoupled physics
        and rendering to keep CPU/GPU utilization low.
        Accepts optional key_callback for interactive GLFW key event handling.
        """
        render_interval = 1.0 / target_fps
        sim_dt = self.model.opt.timestep
        last_render_time = time.time()
        self.has_active_viewer = True

        try:
            with self._lock:
                viewer_cm = mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback)

            with viewer_cm as viewer:
                viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

                while viewer.is_running():
                    now = time.time()
                    self.step()

                    if now - last_render_time >= render_interval:
                        with self._lock:
                            viewer.sync()
                        last_render_time = now

                    elapsed = time.time() - now
                    sleep_time = sim_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        finally:
            self.has_active_viewer = False
