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

> [!TIP]
> Disclaimer: This is a personal project.  The views, code, and opinions expressed here are my own and do not represent those of my current or past employers.

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

The **WAVEGO Pro** uses an elegant mechanical solution: a **closed-loop 4-bar parallel mechanism** with an angled lower shank and foot.

```
                     [Chassis Body]
                           |
                 [Hip Roll Servo / Wiggle]
                           |
             +-------------+-------------+
             |   s = 12.2 mm shaft separation
       (Fore Servo)                (Back Servo)
        Horn A (l_a)                Horn B (l_a)
          \                              /
           \                            /
         Joint P                      Joint Q
            \                          /
             \                        /
       Pushrod\                      / Upper Shank (l_c)
        (l_b)  \                    /
                \                  /
                 \                /
                  \              /
                   \            /
                    \          /
                     [Knee Joint]  <-- <equality><connect> closes loop
                          \
                           \  Lower Shank (l_d)
                            \
                             \  bend angle
                              +-------------O [Foot Contact Sphere]
                                Foot segment (l_e)
```

#### Linkage Dimensions (Matching Official Firmware & CAD)
| Parameter | Length | Description |
| :--- | :--- | :--- |
| **$s$** | $12.2\text{ mm}$ | Fore-to-Back servo horn shaft separation in hip block |
| **$l_a$** | $40.0\text{ mm}$ | Servo horn length (both Fore Horn A and Back Horn B) |
| **$l_b$** | $40.0\text{ mm}$ | Fore pushrod connecting Horn A to the Knee pivot |
| **$l_c$** | $39.8\text{ mm}$ | Upper shank section forming the rear side of the 4-bar loop |
| **$l_d$** | $31.8\text{ mm}$ | Lower shank extending past the Knee pivot |
| **$l_e$** | $30.8\text{ mm}$ | Angled foot segment terminating at the spherical rubber foot |

### Why a Parallel Diamond?
1. **Mass Centralization**: Both the `Fore` (front) and `Back` (rear) servo motors are mounted high up inside the hip assembly, close to the robot's center of mass. The lower leg consists only of lightweight linkages.
2. **Coordinated Motion**:
   - Rotating both horns in opposite directions ($<>$) expands or squashes the diamond, moving the foot vertically up or down.
   - Rotating both horns in unison sweeps the diamond forwards or backwards.
3. **Rigidity & Load Sharing**: Closed kinematic loops distribute mechanical ground impact loads across two parallel linkages rather than concentrating all bending torque on a single motor shaft.

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
│       ├── wavego.xml          # 4-bar MJCF robot definition (joints, links, actuators, camera)
│       └── scene.xml           # Simulation environment (ground plane, lighting, camera)
│
├── app/
│   ├── __init__.py             # Application package marker
│   ├── config.py               # Pydantic settings: OpenRouter API, model, landmarks, server port
│   ├── tools.py                # Typed Google ADK tools (walk_forward, docking, vision, gestures)
│   ├── agent.py                # Google ADK Cognitive Agent using OpenRouter DeepSeek vision
│   ├── navigation.py           # Closed-loop landmark navigation worker, heading controller, stall detection
│   ├── vision_sensor.py        # Passive low-cost frame-differencing motion detector
│   ├── gateway.py              # FastAPI REST, SSE Event Stream Server, and Priority Event Queue
│   └── templates/
│       └── dashboard.html      # Mission Control Web Dashboard (dual cameras, interactive map, cognition feed)
│
├── wavego_protocol.py          # Official Waveshare JSON command constants & packet format
├── robot_interface.py          # Abstract BaseRobot interface & PhysicalRobotController (Serial)
├── wavego_kinematics.py        # Python port of the C++ closed-form IK and gait engine
├── sim.py                      # MuJoCo simulation driver implementing BaseRobot with camera rendering
├── main.py                     # Entrypoint: Starts the robot, motion sensor, and event gateway
├── demo.py                     # Interactive 3D visualizer script for manual evaluation
│
├── pyproject.toml              # Modern Python packaging and dependency config (managed by uv)
├── uv.lock                     # Deterministic dependency lockfile
├── AGENTS.md                   # Operational guidelines & constraints for coding agents
└── README.md                   # This pedagogical guide
```

### How the Architecture Operates

1. **Cognitive Agent & Tools ([`app/agent.py`](app/agent.py), [`app/tools.py`](app/tools.py))**:
   Powered by Google ADK with OpenRouter (`deepseek/deepseek-v4-flash-vision-exp`). The agent has access to granular locomotion tools (`walk_forward`, `walk_backward`, `turn_left`, `turn_right`), home navigation (`navigate_to_landmark`, `return_to_charging_dock`), multimodal vision (`capture_camera_image`), greeting gestures (`perform_handshake`), and speech announcements (`announce_message`).
2. **Autonomous Closed-Loop Navigation Engine ([`app/navigation.py`](app/navigation.py))**:
   Manages closed-loop waypoint navigation between smart home landmarks with proportional heading correction, automatic stall/obstacle detection via odometry delta monitoring, and priority event triggers (`nav_completed`, `nav_stalled`, `nav_feedback`).
3. **Event Ingestion Gateway & Priority Queue ([`app/gateway.py`](app/gateway.py))**:
   FastAPI web service hosting an `asyncio.PriorityQueue` (`P0` Urgent user commands, `P1` High-priority doorbell rings and stall alerts, `P2` Normal navigation feedback, `P3` Background motion). Broadcasts robot telemetry, queue states, and native ADK cognitive events over Server-Sent Events (`/events/stream`).
4. **Passive Vision Motion Sensor ([`app/vision_sensor.py`](app/vision_sensor.py))**:
   Zero-cost local frame differencing running at 2 FPS on the robot's front eye-level camera. Detects scene motion without calling external LLM APIs, firing events only when movement occurs.
5. **Hardware Abstraction Layer ([`robot_interface.py`](robot_interface.py), [`wavego_protocol.py`](wavego_protocol.py))**:
   Universal `BaseRobot` interface compiling commands into official Waveshare JSON packets ($T=110, 111, 112, 113, 114$). Swappable between simulation (`WavegoSim`) and real hardware (`PhysicalRobotController`).
6. **Physics Digital Twin & Kinematics ([`sim.py`](sim.py), [`wavego_kinematics.py`](wavego_kinematics.py))**:
   MuJoCo 500 Hz simulation of the authentic 4-bar linkage mechanism with low-pass servo rate filtering and off-screen camera rendering.

---

## 6. Quickstart & Mission Control Dashboard

### Environment Setup
This repository uses `uv` for fast, isolated Python environment management.

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Configure your OpenRouter API key in `.env`:
   ```bash
   echo "OPENROUTER_API_KEY=your_key_here" > .env
   ```

### Starting the System
Start the robot driver, navigation worker, passive vision motion sensor, and web gateway:
```bash
uv run python main.py
```

### Accessing the Web Dashboard
Open your browser and navigate to:
```
http://localhost:8000/
```

The **Smart Home Mission Control Dashboard** provides a unified real-time operations console:

1. **Dual-Camera Live MJPEG Video Stream**:
   - Toggle between the **Tracking Camera** (3rd-person isometric spectator view) and the **Front Camera** (robot's first-person eye level).
2. **Interactive 2D Landmark Arena Map**:
   - **Full Edge-to-Edge Responsive Canvas**: Automatically resizes to fit your browser or display pane with true 1:1 metric scale.
   - **Height Presets & Draggable Handle**: Toggle between **S** (220px), **M** (340px), and **L** (500px), or drag the bottom-right corner to any custom height.
   - **Click-to-Navigate**: Click directly on any landmark (⚡ **Dock**, 🚪 **Front Door**, 🛋️ **Living Room**, 🍳 **Kitchen**) to dispatch the robot to that location.
   - **Dynamic Odometry Trail & Nav Target**: Visualizes the robot's path across the floor in real-time, with a dashed guide line pointing toward the active destination.
   - **Hover Coordinates**: Real-time metric readout (`X: 1.20m, Y: 0.00m`) follows your mouse.
3. **Stimulus Dispatch Station**:
   - **🔔 Ring Doorbell**: Triggers a simulated visitor event with custom visitor names (e.g. *Alice*).
   - **⚡ Return to Dock**: Sends an urgent command for the robot to return to its home base and crouch onto charging contact plates.
   - **🛑 Emergency Stop**: Immediately cuts motor drive and halts navigation.
   - **Interactive User Prompt Box**: Send natural language requests directly to the agent (e.g., *"Walk forward 0.5 meters and look around"*).
4. **RobotStimulusEvent Priority Queue Visualizer**:
   - Displays real-time **Active Processing** task status with priority badge (`P0`, `P1`, `P2`, `P3`).
   - Live list of waiting events in the priority queue.
   - Collapsible **Audit Log Drawer** recording timestamped history of enqueued, completed, and dropped events.
5. **Real-Time Cognition & Action Feed**:
   - **💭 Agent Thoughts**: Displays internal LLM reasoning and scene interpretation in purple.
   - **🛠️ Action Calls**: Green cards displaying exact tool executions with typed keyword arguments (e.g. `navigate_to_landmark(landmark_name="front_door")`).
   - **📥 Tool Results**: Cyan/slate cards displaying execution feedback.
   - **📢 Spoken Announcements**: Warm amber cards highlighting speech announced through the robot's speaker (`announce_message`).
   - **Auto-Collapsing Cards**: Any entry longer than 2 lines automatically collapses with a `▾ click to expand` toggle, keeping the feed clean and readable.

### External Webhook Triggers (cURL API)
You can also trigger events externally via HTTP:

```bash
# Simulate a smart doorbell ring with facial recognition metadata
curl -X POST http://127.0.0.1:8000/events \
     -H "Content-Type: application/json" \
     -d '{"source": "nest_doorbell", "event_type": "doorbell_press", "payload": {"person_name": "Alice"}}'

# Send a user message / request
curl -X POST http://127.0.0.1:8000/events \
     -H "Content-Type: application/json" \
     -d '{"source": "user_phone", "event_type": "user_message", "payload": {"message": "Please go check the kitchen."}}'
```

### Standalone Kinematics & Visualizer Demo
To test physical gaits, single-leg IK, and keyboard teleoperation directly in the native MuJoCo interactive viewer:
```bash
uv run python demo.py
```
