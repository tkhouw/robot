"""
WAVEGO Pro Interactive Simulation Demo.
Allows real-time interactive testing of individual actions:
- Via 3D viewer keyboard shortcuts (W/A/S/D/Space/C/H/1-4)
- Via terminal CLI commands while viewer is running

Run this script with:
    uv run python demo.py
"""

import sys
import threading
import time
from sim import WavegoSim


def print_help():
    print("\n" + "=" * 65)
    print("        WAVEGO Pro Interactive Controls & Keybindings        ")
    print("=" * 65)
    print(" [Viewer Window Keyboard Shortcuts]")
    print("   W          : Walk Forward (FB = +50)")
    print("   S          : Walk Backward (FB = -50)")
    print("   A          : Turn Left (LR = -50)")
    print("   D          : Turn Right (LR = +50)")
    print("   Space      : Stop walking / Stand upright (T=110)")
    print("   C          : Crouch low (75 mm height)")
    print("   H          : Handshake gesture")
    print("   1, 2, 3, 4 : Test single-leg IK lift for leg 1, 2, 3, or 4")
    print("   [ / ]      : Adjust stand height (-5 mm / +5 mm)")
    print("   T          : Print telemetry (IMU attitude & position)")
    print("   ?          : Print this help menu")
    print("-" * 65)
    print(" [Terminal CLI Commands]")
    print("   walk <fb> <lr>   : e.g. 'walk 50 0' or 'walk 0 30'")
    print("   stand            : Reset to default standing pose")
    print("   crouch           : Crouch low")
    print("   height <h>       : Stand at specific height (e.g. 'height 85')")
    print("   leg <id> <x> <y> <z> : Single leg IK (e.g. 'leg 1 15 85 20')")
    print("   json <json_str>  : Send raw JSON packet (e.g. 'json {\"T\":110}')")
    print("   help             : Print help")
    print("   quit             : Exit demo")
    print("=" * 65 + "\n")


def make_key_callback(robot: WavegoSim):
    """Creates a GLFW key callback for interactive viewer window keypresses."""
    # GLFW keycodes
    GLFW_KEY_W = 87
    GLFW_KEY_S = 83
    GLFW_KEY_A = 65
    GLFW_KEY_D = 68
    GLFW_KEY_SPACE = 32
    GLFW_KEY_C = 67
    GLFW_KEY_H = 72
    GLFW_KEY_T = 84
    GLFW_KEY_1 = 49
    GLFW_KEY_2 = 50
    GLFW_KEY_3 = 51
    GLFW_KEY_4 = 52
    GLFW_KEY_LEFT_BRACKET = 91
    GLFW_KEY_RIGHT_BRACKET = 93
    GLFW_KEY_SLASH = 47

    def on_key(keycode: int):
        if keycode == GLFW_KEY_W:
            print("[KEYBOARD] Walk Forward")
            robot.walk(fb=50.0, lr=0.0)
        elif keycode == GLFW_KEY_S:
            print("[KEYBOARD] Walk Backward")
            robot.walk(fb=-50.0, lr=0.0)
        elif keycode == GLFW_KEY_A:
            print("[KEYBOARD] Turn Left")
            robot.walk(fb=0.0, lr=-50.0)
        elif keycode == GLFW_KEY_D:
            print("[KEYBOARD] Turn Right")
            robot.walk(fb=0.0, lr=50.0)
        elif keycode == GLFW_KEY_SPACE:
            print("[KEYBOARD] Stand / Stop")
            robot.stand()
        elif keycode == GLFW_KEY_C:
            print("[KEYBOARD] Crouch Low")
            robot.crouch()
        elif keycode == GLFW_KEY_H:
            print("[KEYBOARD] Handshake")
            robot.handshake()
        elif keycode in (GLFW_KEY_1, GLFW_KEY_2, GLFW_KEY_3, GLFW_KEY_4):
            leg_id = keycode - GLFW_KEY_1 + 1
            print(f"[KEYBOARD] Lift Leg {leg_id} via IK (y=75mm, x=20mm)")
            robot.set_single_leg(leg_id=leg_id, x=20.0, y=75.0, z=25.0)
        elif keycode == GLFW_KEY_LEFT_BRACKET:
            new_h = max(robot.stand_h - 5.0, 70.0)
            print(f"[KEYBOARD] Lower height -> {new_h:.1f} mm")
            robot.stand_height(new_h)
        elif keycode == GLFW_KEY_RIGHT_BRACKET:
            new_h = min(robot.stand_h + 5.0, 110.0)
            print(f"[KEYBOARD] Raise height -> {new_h:.1f} mm")
            robot.stand_height(new_h)
        elif keycode == GLFW_KEY_T:
            pos, quat = robot.get_pose()
            roll, pitch, yaw = robot.get_imu()
            print(f"[TELEMETRY] Pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) | RPY: ({roll:.1f}°, {pitch:.1f}°, {yaw:.1f}°)")
        elif keycode == GLFW_KEY_SLASH:
            print_help()

    return on_key


def terminal_input_thread(robot: WavegoSim, stop_event: threading.Event):
    """Listens for CLI commands from terminal stdin in background."""
    while not stop_event.is_set():
        try:
            line = sys.stdin.readline()
            if not line:
                break
            cmd = line.strip()
            if not cmd:
                continue

            parts = cmd.split()
            action = parts[0].lower()

            if action in ("quit", "exit", "q"):
                print("Exiting...")
                stop_event.set()
                break
            elif action == "stand":
                print("[CLI] Standing upright")
                robot.stand()
            elif action == "crouch":
                print("[CLI] Crouching low")
                robot.crouch()
            elif action == "walk":
                fb = float(parts[1]) if len(parts) > 1 else 50.0
                lr = float(parts[2]) if len(parts) > 2 else 0.0
                print(f"[CLI] Walking FB={fb}, LR={lr}")
                robot.walk(fb=fb, lr=lr)
            elif action == "stop":
                print("[CLI] Stopping")
                robot.stop()
            elif action == "height":
                h = float(parts[1]) if len(parts) > 1 else 95.0
                print(f"[CLI] Setting height to {h} mm")
                robot.stand_height(h)
            elif action == "leg":
                if len(parts) >= 5:
                    leg_id = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    print(f"[CLI] Single leg IK: leg={leg_id}, x={x}, y={y}, z={z}")
                    robot.set_single_leg(leg_id, x, y, z)
                else:
                    print("Usage: leg <id> <x> <y> <z>")
            elif action == "json":
                json_str = " ".join(parts[1:])
                print(f"[CLI] Sending raw JSON: {json_str}")
                robot.send_cmd(json_str)
            elif action == "help":
                print_help()
            else:
                print(f"Unknown command: '{action}'. Type 'help' for options.")

        except Exception as e:
            print(f"[CLI ERROR] {e}")


if __name__ == "__main__":
    print_help()
    robot = WavegoSim()

    # Start terminal command listener in background
    stop_event = threading.Event()
    cli_thread = threading.Thread(
        target=terminal_input_thread, args=(robot, stop_event), daemon=True
    )
    cli_thread.start()

    # Launch passive viewer with keyboard shortcuts
    key_callback = make_key_callback(robot)
    robot.launch_viewer(key_callback=key_callback)

    stop_event.set()
