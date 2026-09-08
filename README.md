# WAVEGO Pro: Hierarchical Reasoning Architecture & Digital Twin

Welcome to the **WAVEGO Pro** quadruped robotics repository. This project builds a physics-accurate digital twin and high-level reasoning platform for the **Waveshare WAVEGO Pro** 12-DOF quadruped robot.

If you are new to robotics, kinematics, or the **MuJoCo** physics engine, this guide is written as a primer and architectural walkthrough. It covers the core physics concepts, the mechanical design of the robot, the mathematical pipeline, and how the codebase is organized to bridge high-level artificial intelligence with low-level actuator control.

> [!WARNING]
> **Model Fidelity Disclaimer: An Approximation, Not Exact**  
> In MuJoCo, our XML model does **not** match physical photos or factory CAD specifications of the real WAVEGO Pro exactly. During development, several mechanical details had to be estimated or tuned manually—for example, the hip mounting points were initially placed too high on the chassis and had to be adjusted by hand to raise the robot's center of mass, and the 4-bar linkage geometry was approximated to balance contact stability.  
> 
> This model is **not** intended as a photorealistic or manufacturing-grade digital twin. Rather, it is designed to be **"good enough"** to capture the essential kinematics, dynamic balance, and gait cycles required for developing, evaluating, and stress-testing **high-level cognitive planners and reasoning agents** in simulation.

> [!TIP]
> **Attribution & Reference Sources**  
> This repository builds on work from both official and open-source community contributors:
> - **3D STL Meshes & SolidWorks Models**: Sourced from Arnau Crespí's community CAD repository: [arnaucresp0/WAVEGO_3D_MODEL](https://github.com/arnaucresp0/WAVEGO_3D_MODEL).
> - **Official Kinematics & Firmware**: Waveshare's official open-source repository: [waveshareteam/WAVEGO_Pro](https://github.com/waveshareteam/WAVEGO_Pro).
> 
> Our [`wavego_kinematics.py`](wavego_kinematics.py) is a direct, standalone Python port of Waveshare's official `BodyCtrl.cpp` / `BodyCtrl.h` closed-form inverse kinematics and crawl/trot gaits. The upstream C++ code is used solely as an offline reference; no C++ dependencies are required at runtime.

---

## Table of Contents
1. [The Grand Vision: Hierarchical Reasoning Architecture](#1-the-grand-vision-hierarchical-reasoning-architecture)
2. [The Mechanical Design: 4-Bar Parallel Diamond Linkage](#2-the-mechanical-design-4-bar-parallel-diamond-linkage)
3. [The Kinematic Engine: FK, IK, and Gait Cycles](#3-the-kinematic-engine-fk-ik-and-gait-cycles)
4. [The Physics Digital Twin: How MuJoCo Simulates Reality](#4-the-physics-digital-twin-how-mujoco-simulates-reality)
5. [Codebase Anatomy: A Guided Tour](#5-codebase-anatomy-a-guided-tour)
6. [Quickstart & Interactive Viewer](#6-quickstart--interactive-viewer)

---

## 1. The Grand Vision: Hierarchical Reasoning Architecture

Autonomous mobile robotics is fundamentally a problem of **time scales** and **abstraction levels**. 

If a robot tried to use an advanced Large Language Model (LLM) or a deep planning algorithm to calculate the electric currents sent to each servo motor 500 times per second, it would fail: high-level cognitive models are too slow to keep a dynamic quadruped from toppling over under gravity.

Modern autonomous robotics solves this through a **Hierarchical Architecture**:

```
+-------------------------------------------------------------------------+
| Level 3: Cognitive & Reasoning Layer (High-Level AI / Agent Planner)    |
| - Scene understanding, task decomposition, goal planning                |
| - Runs at: 0.1 Hz - 2 Hz (seconds)                                      |
| - Example: "Navigate across the room and inspect the doorway."          |
+-------------------------------------------------------------------------+
                                   |
                                   v  (Dispatches semantic commands: walk(fb, lr), stand(), crouch())
+-------------------------------------------------------------------------+
| Level 2: Kinematic & Gait Generator (Middle-Level Coordination)         |
| - Inverted pendulum balance, foot trajectory curves, closed-form IK     |
| - Runs at: 20 Hz - 50 Hz (milliseconds)                                 |
| - Example: wavego_kinematics.py -> computes 12 target servo angles     |
+-------------------------------------------------------------------------+
                                   |
                                   v  (Sends target joint angles & stiffness)
+-------------------------------------------------------------------------+
| Level 1: Actuator & Physics Layer (Low-Level Dynamics)                  |
| - PD position control, contact friction, closed-loop constraint solver   |
| - Runs at: 500 Hz (microseconds, dt = 0.002s)                          |
| - Example: sim.py (MuJoCo digital twin) / physical SCS0009 bus servos   |
+-------------------------------------------------------------------------+
```

### The Mission of this Repository
This codebase implements **Levels 1 and 2** with mathematical and physical fidelity matching the official physical robot. By establishing an authentic simulation environment and JSON-compatible API, we create a playground where **Level 3 (AI reasoning, planning, and vision)** can safely explore, navigate, and make decisions without risking physical hardware damage.

---

## 2. The Mechanical Design: 4-Bar Parallel Diamond Linkage

Most robotic arms are **serial kinematic chains**: joint 1 connects to link 1, which carries joint 2, which carries link 2. Serial chains are simple to calculate, but they have a severe disadvantage for walking robots: **motors are heavy**. If every joint motor is mounted down on the leg, the leg has high inertia, making fast, agile stepping difficult.

The **WAVEGO Pro** uses an elegant mechanical solution: a **closed-loop 4-bar parallel mechanism**.

```
                [Chassis Torso]
                     |
            [Hip Roll Servo / Wiggle]
                     |
        +------------+------------+
        |                         |
   (Fore Servo)              (Back Servo)
     [Horn A]                  [Horn B]
        \                         /
         \  (Fore Pushrod)       /  (Shank Linkage)
          \                     /
           \                   /
            O=================O  <-- Knee Pivot Vertex
             \               /
              \             /
               \           /
                \         /
                 \       /
                  +-----+
                  |Foot |
```

### Why a Parallel Diamond?
1. **Mass Centralization**: Both the `Fore` (front) and `Back` (rear) servo motors are mounted high up inside the hip assembly, close to the robot's center of mass. The lower leg consists only of lightweight aluminum linkages.
2. **Coordinated Motion**:
   - Rotating both horns in opposite directions ($<>$) expands or squashes the diamond, moving the foot vertically up or down.
   - Rotating both horns in unison sweeps the diamond forwards or backwards.
3. **Rigidity**: Closed kinematic loops distribute mechanical loads between two parallel linkages rather than concentrating all stress on a single motor shaft.

Each leg has **3 degrees of freedom (DOF)**:
1. **Hip Wiggle (Roll)**: Tilts the entire leg outward (abduction) or inward (adduction) to adjust stance width or lean into turns.
2. **Fore Servo (Pitch)**: Drives the anterior horn and pushrod.
3. **Back Servo (Pitch)**: Drives the posterior horn and shank.

With 4 legs, the robot possesses **12 actuated degrees of freedom** in total.

---

## 3. The Kinematic Engine: FK, IK, and Gait Cycles

The math engine is implemented in [`wavego_kinematics.py`](wavego_kinematics.py), which is a pure Python port of Waveshare's official C++ firmware (`BodyCtrl.cpp`).

### Forward Kinematics (FK) vs. Inverse Kinematics (IK)
- **Forward Kinematics (FK)**: Given the 3 servo angles ($\theta_1, \theta_2, \theta_3$), compute the 3D position of the foot $(x, y, z)$.
- **Inverse Kinematics (IK)**: Given where you want the foot to be in 3D space relative to the hip $(x, y, z)$, calculate the exact 3 servo angles required to place the foot there.

For walking, the robot must use **IK**: the gait planner dictates foot paths through space (e.g., "lift the foot $9\text{ mm}$ off the ground and step $25\text{ mm}$ forward"), and the IK engine computes the servo target angles.

### How the WAVEGO IK Works
The WAVEGO IK decouples 3D space into two 2D sub-problems:
1. **Wiggle Plane IK (`wiggle_plane_ik`)**:
   Solves the frontal roll plane ($Y$-$Z$). Given lateral offset $z_{\text{in}}$ and vertical depth $y_{\text{in}}$, it computes the hip roll angle ($\alpha$) and the effective projected leg length.
2. **Single Leg Plane IK (`single_leg_plane_ik` & `simple_linkage_ik`)**:
   Solves the sagittal pitch plane ($X$-$Y$). Using the Law of Cosines on the 4-bar linkage dimensions (linkages $a, b, c, d, e$), it solves for the exact angles of the `Fore` and `Back` servo horns.

### Walking Gaits: Trot vs. Crawl
Walking is a synchronized coordination of 4 legs cycling between two states:
- **Stance Phase**: The foot is firmly planted on the ground, pushing backwards relative to the body to propel the robot forward.
- **Swing Phase**: The foot is lifted into the air and swings forward to prepare for the next step.

The codebase supports two distinct gait algorithms ported from the official firmware:
1. **`simple_gait()` (Diagonal Trot Gait - Default)**:
   Legs move in diagonal pairs (`FL + RR` swing together while `FR + RL` support the body, and vice versa). This is inherently stable because the two supporting feet create an alternating diagonal support axis that aligns with the center of gravity.
2. **`triangular_gait()` (Triangular Crawl Gait)**:
   Three legs remain on the ground forming a support triangle while one leg swings forward. This gait dynamically shifts the robot's center of mass by up to $21\text{ mm}$ sideways (`WALK_MASS_ADJUST`) to keep the center of gravity inside the supporting triangle.

---

## 4. The Physics Digital Twin: How MuJoCo Simulates Reality

To test control software safely, we use **MuJoCo** (Multi-Joint dynamics with Contact), a world-standard physics engine developed by Google DeepMind.

The robot is defined in an **MJCF** (MuJoCo XML Format) model file: [`models/wavego/wavego.xml`](models/wavego/wavego.xml).

### Key Concepts in the MuJoCo Model

#### 1. Closed Kinematic Chains with `<equality><connect>`
MuJoCo represents rigid bodies as open trees (parent-child hierarchies). To simulate the closed 4-bar linkage:
- The `Fore Horn` and `Fore Rod` form one branch of the tree.
- The `Back Horn` and `Shank` form a second branch of the tree.
- A closed kinematic constraint is formed using:
  ```xml
  <equality>
    <connect site1="FL_rod_knee" site2="FL_shank_knee"/>
  </equality>
  ```
  MuJoCo's constraint solver continuously enforces that these two points remain locked together at runtime, creating the authentic parallel linkage behavior.

#### 2. Contact Filtering with `<contact><exclude>`
Because adjacent components meet at joint pivots, their geometric capsules overlap slightly. The `<contact><exclude>` block instructs MuJoCo's collision detector to ignore internal pivot intersections, ensuring that only the spherical rubber feet touch the floor.

#### 3. Position Actuators ($K_p$ and $K_v$)
The 12 servos are modeled as PD position actuators:
$$\tau = K_p (q_{\text{target}} - q) - K_v \dot{q}$$
- $K_p$ (Proportional gain): Represents servo stiffness / holding torque ($60.0\text{ N}\cdot\text{m/rad}$).
- $K_v$ (Derivative gain): Represents velocity damping ($1.5\text{ N}\cdot\text{m}\cdot\text{s/rad}$) to prevent oscillations.

#### 4. Low-Pass Servo Rate Filtering
Physical serial-bus servos (SCS0009) do not jump instantaneously to new positions; their internal microcontrollers ramp motor current smoothly. In [`sim.py`](sim.py), we simulate this hardware behavior using an exponential low-pass filter:
$$q_{\text{filtered}} \leftarrow q_{\text{filtered}} + \alpha (q_{\text{target}} - q_{\text{filtered}}), \quad \alpha = \frac{\Delta t}{\Delta t + \tau_{\text{servo}}}$$
This eliminates unphysical shockwaves and ensures seamless transitions between standing, crouching, and walking.

---

## 5. Codebase Anatomy: A Guided Tour

```
robot/
├── models/
│   └── wavego/
│       ├── assets/             # Official STL 3D mesh files from Waveshare CAD
│       ├── wavego.xml          # 4-bar MJCF robot definition (joints, links, actuators)
│       └── scene.xml           # Simulation environment (ground plane, lighting, camera)
│
├── wavego_kinematics.py        # Python port of the C++ closed-form IK and gait engine
├── sim.py                      # MuJoCo simulation wrapper & official JSON command handler
├── demo.py                     # Interactive 3D visualizer script for manual evaluation
│
├── pyproject.toml              # Modern Python packaging and dependency config (managed by uv)
├── uv.lock                     # Deterministic dependency lockfile
├── AGENTS.md                   # Operational guidelines & constraints for coding agents
└── README.md                   # This pedagogical guide
```

### How the Primary Modules Interact

1. [`demo.py`](demo.py):
   The high-level user entry point. It instantiates `WavegoSim()`, commands the robot to stand or walk, and launches MuJoCo's interactive 3D visualizer.
2. [`sim.py`](sim.py):
   The bridge between software and physics. It maintains the MuJoCo simulation loop, listens for movement commands (e.g. `walk(fb, lr)`), steps the kinematics engine, filters actuator commands, and advances the physics by $\Delta t = 0.002\text{ s}$.
3. [`wavego_kinematics.py`](wavego_kinematics.py):
   The geometric brain. It takes gait cycle phases and computes the exact angles required for all 12 servo joints, mapping them correctly to physical actuator directions.
4. [`models/wavego/wavego.xml`](models/wavego/wavego.xml):
   The physical definition. Defines masses, inertia tensors, STL visual meshes, 4-bar equality constraints, and actuator gains.

---

## 6. Quickstart & Interactive Viewer

### Environment Setup
This repository uses `uv` for fast, isolated Python environment management.

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run the interactive simulation demo:
   ```bash
   uv run python demo.py
   ```

### Interactive Viewer Controls
- **Rotate Camera**: Left-click and drag.
- **Pan Camera**: Right-click and drag.
- **Zoom**: Scroll wheel.
- **Pause / Unpause**: Spacebar.
- **Slow Motion**: Press `S`.
- **Exit**: Close the window or press `ESC`.

### Programmatic Control Example
You can import `WavegoSim` and command the robot in your own scripts:

```python
from sim import WavegoSim
import time

# Initialize simulation and load model
robot = WavegoSim()

# Stand up
robot.stand()

# Walk forward at 50% speed
robot.walk(fb=50.0, lr=0.0)

# Advance physics
for _ in range(1000):
    robot.step()
    time.sleep(0.002)

# Stop walking and return to stand
robot.stop()
```
