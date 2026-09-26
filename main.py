"""
Scout Autonomous Pet Companion Entry Point.
Launches the robot controller, binds ADK tools, starts the passive vision motion detector,
and serves the FastAPI event stream gateway for Google Nest Doorbell and user voice/text commands.
"""

import sys
import threading
import uvicorn
import requests

from sim import QuadrupedSim
from robot_interface import PhysicalRobotController
from app.config import config
from app.tools import set_robot_instance
from app.vision_sensor import PassiveMotionDetector
from app.gateway import app


def on_motion_detected(diff_score: float):
    """Callback from the passive motion sensor running in the background."""
    print(f"\n[PASSIVE VISION DETECTOR] Motion detected (diff={diff_score:.1f})! Firing event...")
    try:
        requests.post(
            f"http://127.0.0.1:{config.port}/events",
            json={
                "source": "motion_sensor",
                "event_type": "motion_detected",
                "payload": {"difference": diff_score},
            },
            timeout=1.0,
        )
    except Exception:
        pass


def main():
    print("=" * 65)
    print("      Scout: Autonomous Pet Companion (Google ADK Architecture)      ")
    print("=" * 65)

    # 1. Select and initialize robot controller
    use_physical = "--real" in sys.argv
    if use_physical:
        serial_port = "/dev/ttyUSB0"
        print(f"Connecting to Physical Scout Quadruped via {serial_port}...")
        robot = PhysicalRobotController(serial_port)
    else:
        print("Initializing Scout Quadruped in MuJoCo Simulation...")
        robot = QuadrupedSim()

    # 2. Bind robot controller to ADK tools
    set_robot_instance(robot)

    # 3. Start passive vision motion detector
    motion_sensor = PassiveMotionDetector(
        robot=robot,
        on_motion_callback=on_motion_detected,
        settling_delay=10.0,
    )
    motion_sensor.start()
    app.state.motion_sensor = motion_sensor
    # 4. Display service information and usage
    print(f"\nModel Backend: {config.model_name}")
    print(f"Interactive Web Dashboard: http://{config.host}:{config.port}/")
    print(f"Event Gateway API:         http://{config.host}:{config.port}/events")
    print(f"Live Video Stream:         http://{config.host}:{config.port}/video/stream?cam=tracking_cam")
    print(f"Live SSE Stream:           http://{config.host}:{config.port}/events/stream")
    print("=" * 65)

    # 5. Serve the FastAPI Gateway & Native 3D Viewer
    # Note: On Linux/X11/Wayland, GLFW requires window creation to occur on the MAIN thread.
    try:
        if hasattr(robot, "launch_viewer"):
            print("\nLaunching native MuJoCo 3D Interactive Window on main thread...")
            server_config = uvicorn.Config(
                app,
                host=config.host,
                port=config.port,
                log_level="info",
                timeout_graceful_shutdown=1,
            )
            server = uvicorn.Server(server_config)
            server_thread = threading.Thread(target=server.run, daemon=True, name="UvicornServerThread")
            server_thread.start()

            # Run interactive 3D viewer on the main thread
            try:
                robot.launch_viewer()
            finally:
                server.should_exit = True
                server_thread.join(timeout=1.5)
        else:
            uvicorn.run(app, host=config.host, port=config.port, timeout_graceful_shutdown=1)
    finally:
        motion_sensor.stop()


if __name__ == "__main__":
    main()


