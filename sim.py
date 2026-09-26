"""
WAVEGO Pro MuJoCo Simulation & High-Level Controller Interface.
Integrates the authentic closed-loop 4-bar parallel linkage mechanism
with the official Waveshare BodyCtrl kinematics and JSON command interface.
"""

import math
import time
import queue
import threading
from typing import Dict, Any, Tuple, Union, List, Optional
import mujoco
import mujoco.viewer
import numpy as np

from wavego_kinematics import (
    WavegoKinematics,
    STAND_HEIGHT,
)

from robot_interface import BaseRobot
from app.config import config
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

        # Motion & Gait State configured from config.py
        self.move_fb = 0.0      # Forward / Backward (-100 to 100)
        self.move_lr = 0.0      # Turn Left / Right (-100 to 100)
        self.walk_cycle = 0.0   # Gait cycle phase [0, 1)
        self.gait_speed = getattr(config, "gait_speed", 2.2)
        self.kinematics.walk_range = getattr(config, "walk_range_mm", 50.0)
        self.kinematics.walk_lift = getattr(config, "walk_lift_mm", 13.0)
        self.stand_h = STAND_HEIGHT

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
        self._render_queue: queue.Queue = queue.Queue()
        self._viewer_thread_id: Optional[int] = None
        self.has_active_viewer: bool = False

        # Cache robot-specific and environment geom IDs for X-Ray silhouette rendering
        self._robot_geom_ids = set()
        self._env_geom_ids = set()
        env_bodies = {
            'world', 'living_sofa', 'coffee_table', 'delivery_box',
            'media_center', 'potted_plant', 'refrigerator', 'dining_table'
        }
        for g_id in range(self.model.ngeom):
            b_id = self.model.geom_bodyid[g_id]
            b_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b_id)
            if b_name and b_name not in env_bodies:
                self._robot_geom_ids.add(g_id)
            else:
                self._env_geom_ids.add(g_id)

        # Battery simulation state
        self.battery_level: float = float(getattr(config, "battery_initial_pct", 100.0))
        self._is_charging: bool = False

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
            self.target_ctrl[:] = self.motion_queue.pop(0)
            target_rad = self.target_ctrl
        else:
            if self.is_moving:
                # Advance gait cycle
                # Speed scales with move_fb magnitude
                speed_factor = max(abs(self.move_fb), abs(self.move_lr)) / 50.0
                speed_factor = min(max(speed_factor, 0.5), 2.0)
                self.walk_cycle = (self.walk_cycle + self.gait_speed * speed_factor * dt) % 1.0

                # Determine turning command (-1: left, 1: right, 0: straight)
                # Lower deadband from 15.0 to 5.0 for responsive closed-loop heading and detour corrections
                if self.move_lr > 5.0:
                    turn_cmd = 1
                elif self.move_lr < -5.0:
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
        """Advances physics, kinematics, and battery simulation by one timestep."""
        if dt is None:
            dt = self.model.opt.timestep
        with self._lock:
            # Battery simulation
            dock_coords = config.landmarks.get("charging_dock", (-0.6, 0.0))
            curr_x, curr_y = float(self.data.qpos[0]), float(self.data.qpos[1])
            dist_to_dock = math.hypot(curr_x - dock_coords[0], curr_y - dock_coords[1])

            # Robot charges if crouched low on the charging dock contact plate
            if dist_to_dock <= 0.25 and self.stand_h <= 75.0:
                self._is_charging = True
                charge_rate = getattr(config, "battery_charge_rate", 10.0)
                self.battery_level = min(100.0, self.battery_level + charge_rate * dt)
            else:
                self._is_charging = False
                if self.is_moving:
                    drain_rate = getattr(config, "battery_drain_rate_moving", 0.35)
                else:
                    drain_rate = getattr(config, "battery_drain_rate_idle", 0.02)
                self.battery_level = max(0.0, self.battery_level - drain_rate * dt)

            # Battery depletion safeguard: shut down locomotion if battery reaches 0%
            if self.battery_level <= 0.0:
                self.move_fb = 0.0
                self.move_lr = 0.0
                self.is_moving = False
                self.stand_h = 70.0  # Collapse onto floor
                self.target_ctrl[:] = self.kinematics.stand(self.stand_h, rad=True)

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
            self.move_fb = 0.0
            self.move_lr = 0.0
            self.stand_h = STAND_HEIGHT
            # Immediately invoke WavegoKinematics to calculate 12-servo standing pose
            self.target_ctrl[:] = self.kinematics.stand(self.stand_h, rad=True)

        elif cmd_type == CMD_STAND_UP_HEIGHT:
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

        elif cmd_type == CMD_BASIC_FUNC:
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

    def get_battery_level(self) -> float:
        """Returns current simulated battery percentage [0.0 - 100.0]."""
        with self._lock:
            return round(float(self.battery_level), 1)

    def is_charging(self) -> bool:
        """Returns True if the robot is currently positioned on dock contacts and replenishing."""
        with self._lock:
            return bool(self._is_charging)

    def _render_camera_raw(self, camera_name: str, width: int, height: int) -> np.ndarray:
        """Internal helper to render RGB array from camera model. Must be called in OpenGL thread."""
        with self._lock:
            renderer = self._get_renderer(width, height)
            renderer.update_scene(self.data, camera=camera_name)
            rgb_array = renderer.render()

            # For top-down spectator camera, guarantee robot visibility with high-visibility outline
            if camera_name == "tracking_cam" and self._robot_geom_ids:
                try:
                    import scipy.ndimage
                    env_indices = list(self._env_geom_ids)
                    orig_alpha = self.model.geom_rgba[env_indices, 3].copy()
                    self.model.geom_rgba[env_indices, 3] = 0.0

                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(self.data, camera=camera_name)
                    seg = renderer.render()
                    renderer.disable_segmentation_rendering()

                    self.model.geom_rgba[env_indices, 3] = orig_alpha

                    robot_mask = np.isin(seg[:, :, 0], list(self._robot_geom_ids))
                    if np.any(robot_mask):
                        dilated = scipy.ndimage.binary_dilation(robot_mask, iterations=2)
                        outline = dilated & ~robot_mask
                        rgb_array = rgb_array.copy()
                        rgb_array[outline] = [0, 235, 255]
                except Exception:
                    pass

            return rgb_array

    def _render_camera_safe(self, camera_name: str, width: int, height: int) -> np.ndarray:
        """
        Thread-safe camera render dispatcher.
        If interactive 3D viewer is running on another thread, dispatches request to viewer thread
        to prevent fatal GLX BadAccess OpenGL context collision.
        """
        current_tid = threading.get_ident()
        if self.has_active_viewer and self._viewer_thread_id is not None and current_tid != self._viewer_thread_id:
            res_queue: queue.Queue = queue.Queue()
            self._render_queue.put((camera_name, width, height, res_queue))
            try:
                res = res_queue.get(timeout=5.0)
                if isinstance(res, Exception):
                    print(f"[SIM WARN] Offscreen camera render failed on viewer thread: {res}")
                    return np.zeros((height, width, 3), dtype=np.uint8)
                return res
            except queue.Empty:
                print(f"[SIM WARN] Camera render timed out for '{camera_name}' (5.0s). Returning fallback frame.")
                return np.zeros((height, width, 3), dtype=np.uint8)
        return self._render_camera_raw(camera_name, width, height)

    def get_camera_image(
        self, width: int = 320, height: int = 240, camera_name: str = "front_camera"
    ) -> bytes:
        """
        Renders an RGB image from the specified camera ('front_camera' or 'tracking_cam')
        and returns PNG encoded bytes.
        """
        import io
        from PIL import Image

        rgb_array = self._render_camera_safe(camera_name, width, height)
        img = Image.fromarray(rgb_array)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def get_disambiguation_views(
        self, width: int = 320, height: int = 240
    ) -> Tuple[bytes, bytes]:
        """
        Captures Left and Right corridor views instantaneously using the front ultra-wide camera lens.
        Renders a single 640x240 wide-angle frame (spanning ~162 deg FOV), then splits and unwarps
        it into Left view (columns looking left) and Right view (columns looking right).
        Zero mechanical delay, zero chassis or neck movement required.
        """
        import io
        from PIL import Image

        wide_frame = self._render_camera_safe("front_camera", width=640, height=height)

        # In MuJoCo optical axes:
        # Camera X-axis points in robot's -Y (right).
        # Therefore:
        # Columns [0 : 320] capture the RIGHT flank/corridor (-60 deg to -80 deg).
        # Columns [320 : 640] capture the LEFT flank/corridor (+60 deg to +80 deg).
        right_crop = wide_frame[:, :320]
        left_crop = wide_frame[:, 320:]

        left_img = Image.fromarray(left_crop).resize((width, height), Image.Resampling.BILINEAR)
        right_img = Image.fromarray(right_crop).resize((width, height), Image.Resampling.BILINEAR)

        buf_l = io.BytesIO()
        left_img.save(buf_l, format="PNG")

        buf_r = io.BytesIO()
        right_img.save(buf_r, format="PNG")

        return buf_l.getvalue(), buf_r.getvalue()

    def get_jpeg_frame(
        self, camera_name: str = "tracking_cam", width: int = 480, height: int = 360, quality: int = 75
    ) -> bytes:
        """
        Fast JPEG rendering for live web streaming over MJPEG.
        Outlines robot when viewing tracking_cam even if occluded by furniture.
        """
        import io
        from PIL import Image

        rgb_array = self._render_camera_safe(camera_name, width, height)
        img = Image.fromarray(rgb_array)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def get_lidar_scan(
        self, num_samples: int = 72, max_range: float = 12.0
    ) -> Dict[str, Any]:
        """
        Simulates a 360-degree DTOF LiDAR scan (Waveshare LD19 standard) using MuJoCo raycasting.
        Sweeps num_samples planar rays around the robot yaw heading from the lidar site / top chassis.
        Returns raw ranges, angular headings in degrees, sector summaries, and minimum obstacle clearance.
        """
        with self._lock:
            # Locate base body id to exclude robot's own bodies and geoms from ray intersections
            base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
            if base_body_id < 0:
                base_body_id = 1

            # Ray origin: lidar site on top chassis
            lidar_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
            if lidar_site_id >= 0:
                origin = self.data.site_xpos[lidar_site_id].copy()
            else:
                origin = self.data.qpos[0:3].copy()
                origin[2] += 0.050

            # Current body yaw heading in radians
            w, x, y, z = self.data.qpos[3:7]
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            body_yaw_rad = math.atan2(siny_cosp, cosy_cosp)

            ranges = []
            angles_deg = []
            geom_id_buf = np.zeros(1, dtype=np.int32)

            # Sweep from -180 to +180 relative to body heading (0 = directly forward)
            # In robot frame: 0° = forward, +90° = left, ±180° = rear, -90° = right
            for i in range(num_samples):
                rel_angle_rad = -math.pi + (2.0 * math.pi * i / num_samples)
                world_angle_rad = body_yaw_rad + rel_angle_rad
                ray_dir = np.array([math.cos(world_angle_rad), math.sin(world_angle_rad), 0.0], dtype=np.float64)

                hit_dist = mujoco.mj_ray(
                    self.model,
                    self.data,
                    origin,
                    ray_dir,
                    None,
                    1,
                    base_body_id,
                    geom_id_buf,
                )

                d = hit_dist if (hit_dist >= 0 and hit_dist <= max_range) else max_range
                ranges.append(round(float(d), 3))
                angles_deg.append(round(float(math.degrees(rel_angle_rad)), 1))

        # Sector aggregations (8 cardinal & intercardinal sectors)
        # Bins: Front [-22.5, 22.5], Front-Left [22.5, 67.5], Left [67.5, 112.5], etc.
        sector_bins = {
            "front": [],
            "front_left": [],
            "left": [],
            "rear_left": [],
            "rear": [],
            "rear_right": [],
            "right": [],
            "front_right": [],
        }

        for dist, deg in zip(ranges, angles_deg):
            if -22.5 <= deg <= 22.5:
                sector_bins["front"].append(dist)
            elif 22.5 < deg <= 67.5:
                sector_bins["front_left"].append(dist)
            elif 67.5 < deg <= 112.5:
                sector_bins["left"].append(dist)
            elif 112.5 < deg <= 157.5:
                sector_bins["rear_left"].append(dist)
            elif deg > 157.5 or deg < -157.5:
                sector_bins["rear"].append(dist)
            elif -157.5 <= deg < -112.5:
                sector_bins["rear_right"].append(dist)
            elif -112.5 <= deg < -67.5:
                sector_bins["right"].append(dist)
            elif -67.5 <= deg < -22.5:
                sector_bins["front_right"].append(dist)

        sectors = {k: round(float(np.min(v)), 2) if v else max_range for k, v in sector_bins.items()}
        min_clearance = round(float(np.min(ranges)), 2) if ranges else max_range

        return {
            "num_samples": num_samples,
            "min_distance_m": min_clearance,
            "sectors": sectors,
            "ranges": ranges,
            "angles_deg": angles_deg,
        }

    def launch_viewer(self, target_fps: int = 30, key_callback=None):
        """
        Launches an interactive 3D visualizer window with decoupled physics
        and rendering to keep CPU/GPU utilization low.
        Accepts optional key_callback for interactive GLFW key event handling.
        """
        render_interval = 1.0 / target_fps
        sim_dt = self.model.opt.timestep
        time_warp = getattr(config, "sim_time_warp", 1.0)
        last_render_time = time.time()
        self.has_active_viewer = True
        self._viewer_thread_id = threading.get_ident()

        try:
            with self._lock:
                viewer_cm = mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback)

            with viewer_cm as viewer:
                viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                # Enable all geom groups in the 3D GUI viewer so all environment assets are visible
                for g in range(6):
                    viewer.opt.geomgroup[g] = 1

                # Center default free-floating MuJoCo camera on the robot
                base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
                if base_body_id >= 0:
                    robot_xpos = self.data.xpos[base_body_id]
                    viewer.cam.lookat[:] = [robot_xpos[0], robot_xpos[1], robot_xpos[2] + 0.05]
                else:
                    viewer.cam.lookat[:] = [-0.60, 0.0, 0.15]
                viewer.cam.distance = 1.6
                viewer.cam.azimuth = -135.0
                viewer.cam.elevation = -20.0

                sim_time_accumulator = 0.0
                last_wall_time = time.perf_counter()

                while viewer.is_running():
                    # Fulfill any offscreen camera render requests safely on OpenGL/viewer thread
                    while not self._render_queue.empty():
                        try:
                            req_cam, req_w, req_h, req_q = self._render_queue.get_nowait()
                            try:
                                frame = self._render_camera_raw(req_cam, req_w, req_h)
                                req_q.put(frame)
                            except Exception as render_err:
                                req_q.put(render_err)
                        except queue.Empty:
                            break

                    now_wall = time.perf_counter()
                    dt_wall = min(now_wall - last_wall_time, 0.1)  # Cap against pause freezes
                    last_wall_time = now_wall
                    sim_time_accumulator += dt_wall * time_warp

                    # Step physics up to wall-clock requirement in batches (Option C)
                    # Eliminates OS sleep quantum bottleneck
                    steps_taken = 0
                    while sim_time_accumulator >= sim_dt and steps_taken < 25:
                        self.step()
                        sim_time_accumulator -= sim_dt
                        steps_taken += 1

                    now = time.time()
                    if now - last_render_time >= render_interval:
                        with self._lock:
                            viewer.sync()
                        last_render_time = now

                    # Sleep briefly to yield CPU without throttling physics rate
                    time.sleep(0.001)
        finally:
            self.has_active_viewer = False
            self._viewer_thread_id = None
