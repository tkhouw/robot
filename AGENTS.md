# Project Guidelines & Agent Rules

## 1. Pre-Write Explanation Rule
The agent must never attempt to create, edit, or overwrite files without first explaining to the user:
1. Exactly what changes or files are being proposed.
2. Why the action is recommended and how it fits into the project goals.

This provides the user with necessary context to review, approve, or reject proposed modifications.

---

## 2. File Editing Tools Rule
- Always use the native `replace_file_content` tool when modifying existing files.
- Do NOT use shell commands (such as `cat`, `sed`, `echo`, or Python one-liners) to edit existing files.
- This ensures the user can clearly inspect and approve changes through visual diffs in the UI.

---

## 3. Python Environment Rules
- **Virtual Environments with `uv`**:
  - Never install dependencies into the system/base Python environment.
  - Always manage environments with `uv` (`uv venv`, `uv add`, `uv pip install`, etc.).
- **Python Version Selection**:
  - If a project already has a `uv` environment/config initialized, assume it uses the correct Python version.
  - When creating a new `uv` environment from scratch, first check installed versions with `uv python list --only-installed` and select the highest installed stable version instead of defaulting to `python`/`python3`.
- **Execution**:
  - Inside a `uv` project or environment, execute scripts with `uv run python ...` and commands with `uv run <command>`.

---

## 4. User Visual Feedback & Efficient Verification Rule
- After making a change to the robot model, kinematics, or simulation parameters, the agent must NEVER run extensive background simulation or testing loops by itself.
- Instead, the agent should directly ask the user to run `uv run python demo.py` and describe what they see in the interactive viewer.
- The user has superior visual comprehension to evaluate mechanical poses, contact, and gaits; relying on user visual feedback conserves context and prevents wasting tokens.

---

## 5. Project Architecture
- **Robot Model**: Authentic WAVEGO Pro quadruped robot with 4-bar parallel closed-loop kinematic leg mechanisms.
- **Model Files**:
  - `models/wavego/wavego.xml`: 4-bar linkage MJCF model with `<equality><connect>` constraints and 12 position servos.
  - `models/wavego/scene.xml`: Scene descriptor with lighting and contact surface.
  - `models/wavego/assets/`: Official WAVEGO STL mesh assets.
- **Kinematics & Control**:
  - `wavego_kinematics.py`: Direct port of Waveshare `BodyCtrl.cpp` closed-form IK, wiggle plane IK, and triangular crawl gait.
  - `sim.py`: MuJoCo simulation interface wrapping 12-servo control and official JSON command handlers.
