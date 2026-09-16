"""
Official WAVEGO Pro Kinematics and Gait Generator.
Ported directly from Waveshare BodyCtrl.cpp and BodyCtrl.h.

Provides exact closed-form planar linkage inverse kinematics,
wiggle plane kinematics, triangular gait generator, and diagonal gait generator.
"""

import math
from typing import List, Tuple


# Structural dimensions from BodyCtrl.cpp (in mm)
LINKAGE_W = 19.15    # Lateral distance between hip wiggle axis and leg plane
LINKAGE_S = 12.20    # Longitudinal spacing between Fore (A) and Back (B) servos
LINKAGE_A = 40.00    # Servo horn length (connecting to servo)
LINKAGE_B = 40.00    # Pushrod length (connecting Fore horn to knee)
LINKAGE_C = 39.8153  # Upper part of leg (Back horn to knee)
LINKAGE_D = 31.7750  # Lower part of leg (knee to ankle)
LINKAGE_E = 30.8076  # Foot offset
WIGGLE_ERROR = 0.0

# Walking parameters from BodyCtrl.cpp (in mm)
WALK_HEIGHT_MAX = 110.0
WALK_HEIGHT_MIN = 75.0
WALK_HEIGHT = 95.0
WALK_LIFT = 9.0
WALK_RANGE = 40.0
WALK_ACC = 5.0
WALK_EXTENDED_X = 16.0
WALK_EXTENDED_Z = 25.0
WALK_MASS_ADJUST = 21.0
STAND_HEIGHT = 95.0
WALK_LIFT_PROP = 0.25


class WavegoKinematics:
    """
    Implements the authentic Waveshare WAVEGO Pro kinematics and gait generation.
    Matches BodyCtrl.cpp joint ordering, coordinate conventions, and linkage geometry.
    """

    def __init__(self):
        # Kinematic geometry
        self.linkage_w = LINKAGE_W
        self.linkage_s = LINKAGE_S
        self.linkage_a = LINKAGE_A
        self.linkage_b = LINKAGE_B
        self.linkage_c = LINKAGE_C
        self.linkage_d = LINKAGE_D
        self.linkage_e = LINKAGE_E
        self.wiggle_error = WIGGLE_ERROR

        # Gait parameters
        self.walk_height_max = WALK_HEIGHT_MAX
        self.walk_height_min = WALK_HEIGHT_MIN
        self.walk_height = WALK_HEIGHT
        self.walk_lift = WALK_LIFT
        self.walk_range = WALK_RANGE
        self.walk_acc = WALK_ACC
        self.walk_extended_x = WALK_EXTENDED_X
        self.walk_extended_z = WALK_EXTENDED_Z
        self.walk_mass_adjust = WALK_MASS_ADJUST
        self.stand_height = STAND_HEIGHT
        self.walk_lift_prop = WALK_LIFT_PROP

        # Precomputed geometric constants matching BodyCtrl.cpp
        self.LA = self.linkage_a
        self.LB = self.linkage_b
        self.LC = self.linkage_c
        self.LD = self.linkage_d
        self.LE = self.linkage_e
        self.LS = self.linkage_s
        self.LW = self.linkage_w

        self.LSs2 = self.LS / 2.0
        self.L_CD = (self.LC + self.LD) ** 2
        self.LAxLA = self.LA ** 2
        self.LBxLB = self.LB ** 2
        self.LWxLW = self.LW ** 2
        self.LExLE = self.LE ** 2
        self.LAxLA_LBxLB = self.LAxLA - self.LBxLB
        self.LBxLB_LAxLA = self.LBxLB - self.LAxLA
        self.LAx2 = 2.0 * self.LA
        self.LBx2 = 2.0 * self.LB
        self.E_PI = 180.0 / math.pi
        self.aLCDE = math.atan((self.LC + self.LD) / self.LE)
        self.sLEDC = math.sqrt(self.LExLE + (self.LD + self.LC) ** 2)

        # 12-servo GoalAngle array (in degrees)
        # Leg 1 (LF): [0] Fore, [1] Back, [2] Hip
        # Leg 2 (LH): [3] Hip,  [4] Fore, [5] Back
        # Leg 3 (RF): [6] Fore, [7] Back, [8] Hip
        # Leg 4 (RH): [9] Hip,  [10] Fore,[11] Back
        self.goal_angles = [0.0] * 12

    def simple_linkage_ik(
        self, LA: float, LB: float, aIn: float, bIn: float
    ) -> Tuple[float, float, float]:
        """
        Closed-form 2-link planar IK for the pushrod and horn.
        Ported from BodyCtrl::simpleLinkageIK.
        """
        if bIn == 0.0:
            val = (self.LAxLA_LBxLB + aIn ** 2) / (self.LAx2 * aIn)
            psi = math.acos(max(min(val, 1.0), -1.0)) * self.E_PI
            alpha = 90.0 - psi
            val2 = (aIn ** 2 + self.LBxLB_LAxLA) / (self.LBx2 * aIn)
            omega = math.acos(max(min(val2, 1.0), -1.0)) * self.E_PI
            beta = psi + omega
        else:
            L2C = aIn ** 2 + bIn ** 2
            LC = math.sqrt(L2C)
            lambda_ = math.atan(bIn / aIn) * 180.0 / math.pi
            val = (self.LAxLA_LBxLB + L2C) / (2.0 * LA * LC)
            psi = math.acos(max(min(val, 1.0), -1.0)) * self.E_PI
            alpha = 90.0 - lambda_ - psi
            val2 = (self.LBxLB_LAxLA + L2C) / (2.0 * LC * LB)
            omega = math.acos(max(min(val2, 1.0), -1.0)) * self.E_PI
            beta = psi + omega
        delta = 90.0 - alpha - beta
        return alpha, beta, delta

    def wiggle_plane_ik(
        self, LA: float, aIn: float, bIn: float
    ) -> Tuple[float, float]:
        """
        Hip wiggle roll plane IK.
        Ported from BodyCtrl::wigglePlaneIK.
        """
        if bIn > 0.0:
            L2C = aIn ** 2 + bIn ** 2
            LC = math.sqrt(L2C)
            lambda_ = math.atan(aIn / bIn) * self.E_PI
            psi = math.acos(max(min(LA / LC, 1.0), -1.0)) * self.E_PI
            LB = math.sqrt(max(L2C - self.LWxLW, 0.0))
            alpha = psi + lambda_ - 90.0
        elif bIn == 0.0:
            alpha = math.asin(max(min(LA / aIn, 1.0), -1.0)) * self.E_PI
            LB = math.sqrt(aIn ** 2)
        else:
            bIn_abs = -bIn
            L2C = aIn ** 2 + bIn_abs ** 2
            LC = math.sqrt(L2C)
            lambda_ = math.atan(aIn / bIn_abs) * self.E_PI
            psi = math.acos(max(min(LA / LC, 1.0), -1.0)) * self.E_PI
            LB = math.sqrt(max(L2C - self.LWxLW, 0.0))
            alpha = 90.0 - lambda_ + psi
        return alpha, LB - self.wiggle_error

    def single_leg_plane_ik(
        self, LS: float, LA: float, LC: float, LD: float, LE: float, xIn: float, yIn: float
    ) -> Tuple[float, float, float]:
        """
        Planar 4-bar leg linkage IK.
        Ported from BodyCtrl::singleLegPlaneIK.
        """
        bufferS = math.sqrt((xIn + self.LSs2) ** 2 + yIn ** 2)
        val = ((xIn + self.LSs2) ** 2 + yIn ** 2 + self.LAxLA - self.L_CD - self.LExLE) / (
            2.0 * bufferS * LA
        )
        lambda_ = math.acos(max(min(val, 1.0), -1.0))
        delta = math.atan((xIn + self.LSs2) / yIn)
        beta = lambda_ - delta
        beta_angle = beta * self.E_PI

        theta = self.aLCDE
        val_omega = (yIn - math.cos(beta) * LA) / self.sLEDC
        omega = math.asin(max(min(val_omega, 1.0), -1.0))
        nu = math.pi - theta - omega
        dFX = math.cos(nu) * LE
        dFY = math.sin(nu) * LE

        mu = math.pi / 2.0 - nu
        dEX = math.cos(mu) * LD
        dEY = math.sin(mu) * LD

        pos_x = xIn + dFX - dEX
        pos_y = yIn - dFY - dEY
        return beta_angle, pos_x, pos_y

    def single_leg_ctrl(self, leg_num: int, x_pos: float, y_pos: float, z_pos: float):
        """
        Computes the 3 servo angles (fore, back, wiggle) for one leg.
        Ported from BodyCtrl::singleLegCtrl.
        LegNum: 1=LF, 2=LH, 3=RF, 4=RH
        """
        if leg_num == 1:
            num_f, num_b, num_w = 0, 1, 2
        elif leg_num == 2:
            num_w, num_f, num_b = 3, 4, 5
        elif leg_num == 3:
            num_f, num_b, num_w = 6, 7, 8
        elif leg_num == 4:
            num_w, num_f, num_b = 9, 10, 11
        else:
            raise ValueError(f"Invalid leg_num {leg_num}, must be 1, 2, 3, or 4")

        wiggle_alpha, wiggle_len = self.wiggle_plane_ik(self.linkage_w, z_pos, y_pos)
        alpha_out, x_buf, y_buf = self.single_leg_plane_ik(
            self.linkage_s,
            self.linkage_a,
            self.linkage_c,
            self.linkage_d,
            self.linkage_e,
            x_pos,
            wiggle_len,
        )
        beta_out, _, _ = self.simple_linkage_ik(
            self.linkage_a, self.linkage_b, y_buf, x_buf - self.LSs2
        )

        self.goal_angles[num_w] = wiggle_alpha
        self.goal_angles[num_f] = 90.0 - beta_out
        self.goal_angles[num_b] = alpha_out

    def single_gait_ctrl(
        self,
        leg_num: int,
        status_input: float,
        cycle_input: float,
        direction_input: float,
        extended_x: float,
        extended_z: float,
    ):
        """
        Generates foot trajectory and solves IK for one leg at phase cycle_input.
        Ported from BodyCtrl::singleGaitCtrl.
        """
        r_direction = direction_input * math.pi / 180.0
        acc = self.walk_acc
        rng = self.walk_range * status_input
        tot_dist = acc * 2.0 + rng
        o_wlp = 1.0 - self.walk_lift_prop

        if cycle_input < o_wlp:
            # Stance phase
            t1 = (acc / tot_dist) * o_wlp
            t2 = ((acc + rng) / tot_dist) * o_wlp
            if cycle_input <= t1:
                y_gait = (
                    (self.walk_height - self.walk_lift)
                    + (cycle_input / t1) * self.walk_lift
                )
            elif cycle_input <= t2:
                y_gait = self.walk_height
            else:
                y_gait = self.walk_height - (
                    (cycle_input - t2) / (o_wlp - t2)
                ) * self.walk_lift
            r_dist = (rng / 2.0 + acc) - (cycle_input / o_wlp) * tot_dist
        else:
            # Swing phase
            y_gait = self.walk_height - self.walk_lift
            r_dist = -(rng / 2.0 + acc) + (
                (cycle_input - o_wlp) / self.walk_lift_prop
            ) * tot_dist

        x_gait = math.cos(r_direction) * r_dist
        z_gait = math.sin(r_direction) * r_dist
        self.single_leg_ctrl(
            leg_num, x_gait + extended_x, y_gait, z_gait + extended_z
        )

    def simple_gait(
        self, global_input: float, direction_angle: float = 0.0, turn_cmd: int = 0, rad: bool = True
    ) -> List[float]:
        """
        Diagonal trot gait.
        Ported from BodyCtrl::simpleGait.
        Returns joint angles in radians by default (rad=True) or degrees (rad=False).
        """
        group_a = global_input % 1.0
        group_b = (global_input + 0.5) % 1.0

        if turn_cmd == 0:
            self.single_gait_ctrl(1, 1.0, group_a,  direction_angle,  self.walk_extended_x, self.walk_extended_z)
            self.single_gait_ctrl(4, 1.0, group_a, -direction_angle, -self.walk_extended_x, self.walk_extended_z)
            self.single_gait_ctrl(2, 1.0, group_b,  direction_angle, -self.walk_extended_x, self.walk_extended_z)
            self.single_gait_ctrl(3, 1.0, group_b, -direction_angle,  self.walk_extended_x, self.walk_extended_z)
        elif turn_cmd == -1:  # Left
            self.single_gait_ctrl(1, 0.6, group_a,  90.0,  self.walk_extended_x, self.walk_extended_z)
            self.single_gait_ctrl(4, 0.6, group_a,  90.0, -self.walk_extended_x, self.walk_extended_z)
            self.single_gait_ctrl(2, 0.6, group_b, -90.0, -self.walk_extended_x, self.walk_extended_z)
            self.single_gait_ctrl(3, 0.6, group_b, -90.0,  self.walk_extended_x, self.walk_extended_z)
        elif turn_cmd == 1:   # Right
            self.single_gait_ctrl(1, 0.6, group_a, -90.0,  self.walk_extended_x, self.walk_extended_z)
            self.single_gait_ctrl(4, 0.6, group_a, -90.0, -self.walk_extended_x, self.walk_extended_z)
            self.single_gait_ctrl(2, 0.6, group_b,  90.0, -self.walk_extended_x, self.walk_extended_z)
            self.single_gait_ctrl(3, 0.6, group_b,  90.0,  self.walk_extended_x, self.walk_extended_z)

        return self.get_joint_angles_rad(self.goal_angles) if rad else list(self.goal_angles)

    def triangular_gait(
        self, global_input: float, direction_angle: float = 0.0, turn_cmd: int = 0, rad: bool = True
    ) -> List[float]:
        """
        4-phase triangular crawl gait with dynamic center-of-mass shift.
        Ported from BodyCtrl::triangularGait.
        Returns joint angles in radians by default (rad=True) or degrees (rad=False).
        """
        step_b = global_input % 1.0
        step_c = (global_input + 0.25) % 1.0
        step_d = (global_input + 0.50) % 1.0
        step_a = (global_input + 0.75) % 1.0

        # Dynamic center-of-mass adjustment
        if global_input <= 0.25:
            prop = global_input
            a_input = self.walk_mass_adjust - (prop / 0.125) * self.walk_mass_adjust
            b_input = -self.walk_mass_adjust
        elif global_input <= 0.50:
            prop = global_input - 0.25
            a_input = -self.walk_mass_adjust + (prop / 0.125) * self.walk_mass_adjust
            b_input = -self.walk_mass_adjust + (prop / 0.125) * self.walk_mass_adjust
        elif global_input <= 0.75:
            prop = global_input - 0.50
            a_input = self.walk_mass_adjust - (prop / 0.125) * self.walk_mass_adjust
            b_input = self.walk_mass_adjust
        else:
            prop = global_input - 0.75
            a_input = -self.walk_mass_adjust + (prop / 0.125) * self.walk_mass_adjust
            b_input = self.walk_mass_adjust - (prop / 0.125) * self.walk_mass_adjust

        if turn_cmd == 0:
            self.single_gait_ctrl(1, 1.0, step_a,  direction_angle,  self.walk_extended_x - a_input, self.walk_extended_z - b_input)
            self.single_gait_ctrl(4, 1.0, step_d, -direction_angle, -self.walk_extended_x - a_input, self.walk_extended_z + b_input)
            self.single_gait_ctrl(2, 1.0, step_b,  direction_angle, -self.walk_extended_x - a_input, self.walk_extended_z - b_input)
            self.single_gait_ctrl(3, 1.0, step_c, -direction_angle,  self.walk_extended_x - a_input, self.walk_extended_z + b_input)
        elif turn_cmd == -1:
            self.single_gait_ctrl(1, 0.6, step_a,  90.0,  self.walk_extended_x - a_input, self.walk_extended_z - b_input)
            self.single_gait_ctrl(4, 0.6, step_d,  90.0, -self.walk_extended_x - a_input, self.walk_extended_z + b_input)
            self.single_gait_ctrl(2, 0.6, step_b, -90.0, -self.walk_extended_x - a_input, self.walk_extended_z - b_input)
            self.single_gait_ctrl(3, 0.6, step_c, -90.0,  self.walk_extended_x - a_input, self.walk_extended_z + b_input)
        elif turn_cmd == 1:
            self.single_gait_ctrl(1, 0.6, step_a, -90.0,  self.walk_extended_x - a_input, self.walk_extended_z - b_input)
            self.single_gait_ctrl(4, 0.6, step_d, -90.0, -self.walk_extended_x - a_input, self.walk_extended_z + b_input)
            self.single_gait_ctrl(2, 0.6, step_b,  90.0, -self.walk_extended_x - a_input, self.walk_extended_z - b_input)
            self.single_gait_ctrl(3, 0.6, step_c,  90.0,  self.walk_extended_x - a_input, self.walk_extended_z + b_input)

        return self.get_joint_angles_rad(self.goal_angles) if rad else list(self.goal_angles)

    def stand(self, height: float = STAND_HEIGHT, rad: bool = True) -> List[float]:
        """
        Computes 12 servo angles for default standing pose at specified height.
        Ported from BodyCtrl::standUp.
        Returns joint angles in radians by default (rad=True) or degrees (rad=False).
        """
        self.single_leg_ctrl(1,  self.walk_extended_x, height, self.walk_extended_z)
        self.single_leg_ctrl(2, -self.walk_extended_x, height, self.walk_extended_z)
        self.single_leg_ctrl(3,  self.walk_extended_x, height, self.walk_extended_z)
        self.single_leg_ctrl(4, -self.walk_extended_x, height, self.walk_extended_z)
        return self.get_joint_angles_rad(self.goal_angles) if rad else list(self.goal_angles)

    @staticmethod
    def bessel_ctrl(num_start: float, num_end: float, rate_input: float) -> float:
        """
        Smooth cosine interpolation between num_start and num_end.
        Ported directly from BodyCtrl::besselCtrl.
        """
        return (num_end - num_start) * ((math.cos(rate_input * math.pi - math.pi) + 1.0) / 2.0) + num_start

    def generate_handshake_trajectory(self) -> List[List[float]]:
        """
        Generates the complete sequence of 12-joint radian setpoints for the
        official Waveshare handshake greeting gesture.
        Ported directly from BodyCtrl::functionHandshake().
        """
        trajectory: List[List[float]] = []

        # Phase 1: Sit back onto rear legs, lean body back, and stabilize base
        for step in range(0, 101, 2):
            i = step / 100.0
            self.single_leg_ctrl(
                1,
                self.bessel_ctrl(self.walk_extended_x, 0.0, i),
                self.bessel_ctrl(self.walk_height, self.walk_height_max, i),
                self.bessel_ctrl(self.walk_extended_z, -15.0, i),
            )
            self.single_leg_ctrl(
                3,
                self.bessel_ctrl(self.walk_extended_x, 0.0, i),
                self.bessel_ctrl(self.walk_height, self.walk_height_max, i),
                self.walk_extended_z,
            )
            self.single_leg_ctrl(
                2,
                -self.walk_extended_x,
                self.bessel_ctrl(self.walk_height, self.walk_height_min - 10.0, i),
                self.bessel_ctrl(self.walk_extended_z, 2.0 * self.walk_extended_z, i),
            )
            self.single_leg_ctrl(
                4,
                -self.walk_extended_x,
                self.bessel_ctrl(self.walk_height, self.walk_height_min - 10.0, i),
                self.bessel_ctrl(self.walk_extended_z, 2.0 * self.walk_extended_z, i),
            )
            trajectory.append(self.get_joint_angles_rad())

        # Phase 2: Lift front right paw (Leg 3) up and extend forward
        for step in range(0, 101, 2):
            i = step / 100.0
            self.single_leg_ctrl(
                3,
                self.bessel_ctrl(0.0, self.walk_range / 2.0 + self.walk_extended_x, i),
                self.bessel_ctrl(self.walk_height_max, self.walk_height_min, i),
                self.bessel_ctrl(self.walk_extended_z, 0.0, i),
            )
            trajectory.append(self.get_joint_angles_rad())

        # Phase 3: Paw shake oscillation (3 shakes up and down)
        for _ in range(3):
            for step in range(0, 101, 4):
                i = step / 100.0
                self.single_leg_ctrl(
                    3,
                    self.walk_range / 2.0 + self.walk_extended_x,
                    self.bessel_ctrl(self.walk_height_min, self.walk_height_min + 30.0, i),
                    0.0,
                )
                trajectory.append(self.get_joint_angles_rad())
            for step in range(0, 101, 4):
                i = step / 100.0
                self.single_leg_ctrl(
                    3,
                    self.walk_range / 2.0 + self.walk_extended_x,
                    self.bessel_ctrl(self.walk_height_min + 30.0, self.walk_height_min, i),
                    0.0,
                )
                trajectory.append(self.get_joint_angles_rad())

        # Phase 4: Return paw to ground and restore upright standing stance
        for step in range(0, 101, 2):
            i = step / 100.0
            self.single_leg_ctrl(
                1,
                self.bessel_ctrl(0.0, self.walk_extended_x, i),
                self.bessel_ctrl(self.walk_height_max, self.walk_height, i),
                self.bessel_ctrl(-15.0, self.walk_extended_z, i),
            )
            self.single_leg_ctrl(
                3,
                self.bessel_ctrl(self.walk_range / 2.0 + self.walk_extended_x, self.walk_extended_x, i),
                self.bessel_ctrl(self.walk_height_min, self.walk_height, i),
                self.bessel_ctrl(0.0, self.walk_extended_z, i),
            )
            self.single_leg_ctrl(
                2,
                -self.walk_extended_x,
                self.bessel_ctrl(self.walk_height_min - 10.0, self.walk_height, i),
                self.bessel_ctrl(2.0 * self.walk_extended_z, self.walk_extended_z, i),
            )
            self.single_leg_ctrl(
                4,
                -self.walk_extended_x,
                self.bessel_ctrl(self.walk_height_min - 10.0, self.walk_height, i),
                self.bessel_ctrl(2.0 * self.walk_extended_z, self.walk_extended_z, i),
            )
            trajectory.append(self.get_joint_angles_rad())

        return trajectory

    def get_joint_angles_rad(self, angles_deg: List[float] = None) -> List[float]:
        """
        Converts degrees to radians for all 12 joints, applying physical servo directions
        matching official Waveshare ServoDirection and MuJoCo joint rotation axes:
        - Fore horn pitch (indices 0, 4, 6, 10): negative radian (rotates forward)
        - Back horn pitch (indices 1, 5, 7, 11): positive radian (rotates backward)
        - Hip wiggle roll (indices 2, 3, 8, 9): negative radian (outward abduction)
        """
        if angles_deg is None:
            angles_deg = self.goal_angles

        rad_list = [0.0] * 12
        # Fore horn pitch joints: rotate forward in MuJoCo
        for i in [0, 4, 6, 10]:
            rad_list[i] = -angles_deg[i] * math.pi / 180.0
        # Back horn pitch joints: rotate backward in MuJoCo
        for i in [1, 5, 7, 11]:
            rad_list[i] = angles_deg[i] * math.pi / 180.0
        # Hip wiggle roll joints: outward abduction (+radian rotates outward for both left and right frames)
        for i in [2, 3, 8, 9]:
            rad_list[i] = angles_deg[i] * math.pi / 180.0

        return rad_list
