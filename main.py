"""
WAVEGO Pro Cognitive Agent Entry Point.
Launches the robot controller, binds ADK tools, starts the passive vision motion detector,
and serves the FastAPI event stream gateway for Google Nest Doorbell and user voice/text commands.
"""

import sys
import threading
import uvicorn
import requests

from sim import WavegoSim
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
            f"http://127.0.0.1:{config.port}/events/trigger",
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
    print("   WAVEGO Pro Cognitive Agent: Google ADK Event-Driven System   ")
    print("=" * 65)

    # 1. Select and initialize robot controller
    use_physical = "--real" in sys.argv
    if use_physical:
        serial_port = "/dev/ttyUSB0"
        print(f"Connecting to Physical WAVEGO Pro via {serial_port}...")
        robot = PhysicalRobotController(serial_port)
    else:
        print("Initializing WAVEGO Pro in MuJoCo Simulation...")
        robot = WavegoSim()

    # 2. Bind robot controller to ADK tools
    set_robot_instance(robot)

    # 3. Start passive vision motion detector
    motion_sensor = PassiveMotionDetector(robot=robot, on_motion_callback=on_motion_detected)
    motion_sensor.start()
    print("Passive Vision Motion Detector: ACTIVE (monitoring for frame changes)")

    # 4. Display service information and usage
    print(f"\nModel Backend: {config.model_name}")
    print(f"Event Gateway: http://{config.host}:{config.port}")
    print("\nSupported External Event Endpoints:")
    print(f" - POST http://127.0.0.1:{config.port}/events/trigger  (Nest Doorbell / Sensors)")
    print(f" - POST http://127.0.0.1:{config.port}/events/command  (Phone Voice / Text Commands)")
    print(f" - GET  http://127.0.0.1:{config.port}/events/stream   (Live SSE Thought & Action Stream)")
    print("=" * 65)

    # 5. Serve the FastAPI Gateway
    try:
        uvicorn.run(app, host=config.host, port=config.port)
    finally:
        motion_sensor.stop()


if __name__ == "__main__":
    main()

