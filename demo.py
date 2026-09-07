"""
WAVEGO Pro Interactive Simulation Demo.
Run this script with:
    uv run python demo.py
"""

from sim import WavegoSim

if __name__ == "__main__":
    print("Launching WAVEGO Pro MuJoCo Simulation...")
    print("Commands:")
    print("  robot.stand()      - Stands up")
    print("  robot.walk(fb, lr) - Walks forward/backward and turns")
    print("  robot.crouch()     - Crouches low")
    print("Press ESC or close the viewer window to exit.")

    robot = WavegoSim()
    robot.walk()
    robot.launch_viewer()
