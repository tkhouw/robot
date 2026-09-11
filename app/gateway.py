"""
FastAPI Event Stream Gateway for the WAVEGO Pro Quadruped.
Exposes endpoints for external event triggers (Nest Doorbell, Vision Motion)
and voice/text commands, streaming agent reasoning via Server-Sent Events (SSE).
"""

import asyncio
import functools
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

from app.agent import QuadrupedCognitiveRuntime
from app.config import config
from app.tools import get_robot, get_nav_manager

app = FastAPI(title="WAVEGO Pro Cognitive Agent Gateway")

# Dedicated single-thread executor for OpenGL/GLX rendering to avoid X11 BadAccess multi-thread conflicts
render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="MuJoCoRenderer")

# Enable CORS for phone/laptop web controllers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Runtime & Event Queues
runtime: QuadrupedCognitiveRuntime = None
# Bounded PriorityQueue (maxsize=15) storing tuples: (priority: int, counter: int, prompt_text: str, event_meta: dict)
event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=15)
sse_subscribers: List[asyncio.Queue] = []
is_agent_busy: bool = False
active_task: Optional[Dict[str, Any]] = None
recent_event_audit: List[Dict[str, Any]] = []  # Ring buffer of recent ingested/dropped events
_queue_counter: int = 0  # Tie-breaker for PriorityQueue

# Debounce tracking tables (event_key -> timestamp of last accepted event)
_last_event_timestamps: Dict[str, float] = {}
DEBOUNCE_WINDOWS = {
    "doorbell_press": 60.0,   # 1 minute debounce
    "motion_detected": 30.0,  # 30 seconds debounce
}


def log_event_audit(event_type: str, source: str, priority: int, status: str, detail: str = ""):
    """Logs recent event activity to the audit ring buffer (max 15 items)."""
    global recent_event_audit
    audit_entry = {
        "timestamp": time.time(),
        "event_type": event_type,
        "source": source,
        "priority": priority,
        "status": status,  # "enqueued", "completed", "dropped"
        "detail": detail,
    }
    recent_event_audit.insert(0, audit_entry)
    recent_event_audit = recent_event_audit[:15]


async def broadcast_queue_state():
    """Broadcasts current snapshot of active task, queued events, and recent audit log."""
    items = []
    # Peek into PriorityQueue internal heap without removing items
    try:
        raw_items = list(event_queue._queue)
        for item in sorted(raw_items, key=lambda x: (x[0], x[1])):
            priority, counter, prompt, meta = item
            items.append({
                "priority": priority,
                "counter": counter,
                "prompt": prompt,
                "source": meta.get("source", "unknown"),
                "event_type": meta.get("event_type", "unknown"),
                "timestamp": meta.get("timestamp", time.time()),
            })
    except Exception:
        pass

    state_payload = {
        "is_busy": is_agent_busy,
        "queue_size": event_queue.qsize(),
        "active_task": active_task,
        "queued_events": items,
        "recent_audit": recent_event_audit,
    }
    await broadcast_sse("queue_state", state_payload)


class RobotStimulusEvent(BaseModel):
    source: str  # e.g. "user_phone", "nest_doorbell", "motion_sensor"
    event_type: str  # e.g. "user_message", "doorbell_press", "motion_detected"
    payload: Dict[str, Any] = {}


def get_event_priority(event: RobotStimulusEvent) -> int:
    """
    Returns numeric priority (lower value = higher priority).
    0: user_message (Immediate human command/query)
    1: doorbell_press, nav_completed, nav_stalled (High-priority action triggers & arrival alerts)
    2: nav_feedback (Normal navigation progress feedback)
    3: motion_detected, passive camera alert, generic sensor events
    """
    if event.event_type == "user_message":
        return 0
    if event.event_type in ("doorbell_press", "nav_completed", "nav_stalled"):
        return 1
    if event.event_type == "nav_feedback":
        return 2
    if event.event_type == "motion_detected":
        return 3
    return 3


def check_debounce(event: RobotStimulusEvent) -> bool:
    """Returns True if the event should be processed, or False if debounced/dropped."""
    debounce_duration = DEBOUNCE_WINDOWS.get(event.event_type)
    if not debounce_duration:
        return True

    now = time.time()
    key = f"{event.source}:{event.event_type}"
    last_time = _last_event_timestamps.get(key, 0.0)

    if (now - last_time) < debounce_duration:
        return False

    _last_event_timestamps[key] = now
    return True


async def broadcast_sse(event_type: str, data: Any):
    """Pushes a message to all active SSE browser connections."""
    message = json.dumps({"type": event_type, "data": data, "timestamp": time.time()})
    for q in list(sse_subscribers):
        try:
            await q.put(message)
        except Exception:
            pass


async def event_processor_loop():
    """Background consumer worker processing incoming stimulus events sequentially by priority."""
    global is_agent_busy, active_task
    while True:
        priority, counter, prompt_text, meta = await event_queue.get()
        is_agent_busy = True
        active_task = {
            "priority": priority,
            "counter": counter,
            "prompt": prompt_text,
            "source": meta.get("source", "unknown"),
            "event_type": meta.get("event_type", "unknown"),
            "start_time": time.time(),
        }

        # Toggle motion suppression on passive vision sensor while agent is reasoning/acting
        motion_sensor = getattr(app.state, "motion_sensor", None)
        if motion_sensor:
            motion_sensor.suppress_motion_detection = True

        await broadcast_sse("agent_turn_start", {"priority": priority, "prompt": prompt_text})
        await broadcast_queue_state()

        try:
            async for adk_event in runtime.execute_prompt(prompt_text):
                # Serialize the native Google ADK Event into a structured dictionary
                event_dict = adk_event.model_dump(mode="json", exclude_none=True)

                # Extract human-readable summary for console logging
                if adk_event.actions and getattr(adk_event.actions, "tool_calls", None):
                    for tc in adk_event.actions.tool_calls:
                        print(f"[ADK TOOL CALL]: {tc.name}({tc.args})")

                if adk_event.content and getattr(adk_event.content, "parts", None):
                    for part in adk_event.content.parts:
                        if hasattr(part, "text") and part.text:
                            print(f"[ADK AGENT THOUGHT/RESPONSE]: {part.text}")

                await broadcast_sse("agent_adk_event", event_dict)

        except Exception as e:
            print(f"[AGENT ERROR]: {e}")
            await broadcast_sse("agent_error", {"error": str(e)})
        finally:
            log_event_audit(
                meta.get("event_type", "task"),
                meta.get("source", "system"),
                priority,
                "completed",
                detail=f"Duration: {round(time.time() - active_task['start_time'], 1)}s" if active_task else "",
            )
            is_agent_busy = False
            active_task = None
            if motion_sensor:
                motion_sensor.suppress_motion_detection = False
            await broadcast_sse("agent_turn_complete", {"status": "idle"})
            event_queue.task_done()
            await broadcast_queue_state()


is_shutting_down: bool = False

async def telemetry_broadcast_loop():
    """Pushes real-time telemetry over the existing SSE stream (4 Hz) without polling."""
    while not is_shutting_down:
        try:
            if sse_subscribers:
                robot = get_robot()
                pos, _ = robot.get_pose()
                rpy = robot.get_imu()
                nav = get_nav_manager()
                motion_sensor = getattr(app.state, "motion_sensor", None)

                data = {
                    "pos": [round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)],
                    "yaw": round(float(rpy[2]), 1),
                    "nav_target": nav.target_name if nav.is_navigating else None,
                    "nav_distance": round(nav.remaining_distance, 2) if nav.is_navigating else 0.0,
                    "nav_progress": nav.progress_pct if nav.is_navigating else 0.0,
                    "nav_eta": nav.eta_seconds if nav.is_navigating else 0.0,
                    "is_busy": is_agent_busy,
                    "motion_suppressed": getattr(motion_sensor, "suppress_motion_detection", False) if motion_sensor else False,
                }
                await broadcast_sse("telemetry", data)
        except Exception:
            pass
        await asyncio.sleep(0.25)


@app.on_event("startup")
async def startup_event():
    global runtime
    runtime = QuadrupedCognitiveRuntime()
    asyncio.create_task(event_processor_loop())
    asyncio.create_task(telemetry_broadcast_loop())


@app.on_event("shutdown")
async def shutdown_event():
    """Notifies active UI clients of shutdown and unblocks all streaming generators."""
    global is_shutting_down
    is_shutting_down = True
    # Notify clients
    await broadcast_sse("server_shutdown", {"message": "Server stopped by user."})
    await asyncio.sleep(0.05)  # Yield to allow network buffers to flush before closing sockets
    # Unblock all waiting queue.get() calls
    for q in sse_subscribers:
        await q.put(None)
    render_executor.shutdown(wait=False)


@app.post("/events")
async def receive_event(event: RobotStimulusEvent):
    """
    Unified ingestion endpoint with Priority Queue, debouncing, and rate limiting.
    """
    global _queue_counter
    priority = get_event_priority(event)

    # 1. Debounce check (e.g. 1-min for doorbell, 30s for motion)
    if not check_debounce(event):
        log_event_audit(event.event_type, event.source, priority, "dropped", detail="Debounced (rate limit window active)")
        await broadcast_queue_state()
        return {
            "status": "dropped",
            "reason": f"debounced (rate limited to once per {DEBOUNCE_WINDOWS.get(event.event_type)}s)",
        }

    # 2. Busy State Rate Limiting
    # Protect priority 0 (user) and priority 1 (doorbell, nav_completed, nav_stalled) from being dropped!
    if is_agent_busy and priority >= 3:
        log_event_audit(event.event_type, event.source, priority, "dropped", detail="Robot actively busy with higher priority task")
        await broadcast_queue_state()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Robot is currently busy executing an active task. Low-priority sensor alert dropped.",
        )

    # 3. Queue Full Check: Never reject priority 0 or priority 1
    if event_queue.full():
        if priority > 1:
            log_event_audit(event.event_type, event.source, priority, "dropped", detail="Event queue full (capacity 15)")
            await broadcast_queue_state()
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Robot event queue is full. Lower-priority event dropped.",
            )

    # 4. Format prompt
    if event.event_type == "user_message":
        msg = event.payload.get("message") or event.payload.get("text") or ""
        agent_prompt = f"[USER MESSAGE from {event.source}]: {msg}"

    elif event.event_type == "nav_feedback":
        landmark = event.payload.get("landmark", "destination")
        pct = event.payload.get("progress_pct", 0.0)
        dist = event.payload.get("distance_remaining_m", 0.0)
        eta = event.payload.get("eta_seconds", 0.0)
        agent_prompt = (
            f"[SYSTEM EVENT: NAVIGATION PROGRESS] En route to '{landmark}': "
            f"{pct:.1f}% complete, {dist:.2f}m remaining, estimated ETA: {eta:.1f}s."
        )
        # Also broadcast progress directly to dashboard UI widget
        await broadcast_sse("nav_progress", event.payload)

    elif event.event_type == "nav_completed":
        landmark = event.payload.get("landmark", "destination")
        final_pos = event.payload.get("final_pos", [0, 0])
        agent_prompt = (
            f"[SYSTEM EVENT: NAVIGATION COMPLETED] The robot has arrived at destination '{landmark}' "
            f"(position: {final_pos}). Proceed with your next task (e.g. greeting visitor or inspecting area)."
        )

    elif event.event_type == "nav_stalled":
        landmark = event.payload.get("landmark", "destination")
        dist = event.payload.get("distance_remaining_m", 0.0)
        agent_prompt = (
            f"[SYSTEM ALERT: NAVIGATION STALLED] Robot forward progress toward '{landmark}' has stalled "
            f"({dist:.2f}m remaining). A physical obstacle or wall was encountered! "
            f"Locomotion has been safely halted. Please call `capture_camera_image` to inspect the blockage, "
            f"and use `walk_backward` or turning to navigate around it."
        )

    elif event.event_type == "nav_cancelled":
        landmark = event.payload.get("landmark", "destination")
        agent_prompt = f"[SYSTEM EVENT: NAVIGATION CANCELLED] Navigation to '{landmark}' was cancelled. Robot is safely standing."

    elif event.source == "nest_doorbell" and event.event_type == "doorbell_press":
        person_name = event.payload.get("person_name")
        if person_name:
            agent_prompt = (
                f"[SYSTEM EVENT: DOORBELL RING] The Google Home Nest doorbell has just rung! "
                f"Familiar visitor '{person_name}' is at the front door. "
                f"Please navigate to the front door and announce a polite, personalized greeting to {person_name}."
            )
        else:
            agent_prompt = (
                "[SYSTEM EVENT: DOORBELL RING] The Google Home Nest doorbell has just rung! "
                "An unrecognized visitor is at the front door. "
                "Please navigate to the front door and announce a polite greeting."
            )

    elif event.source == "motion_sensor" and event.event_type == "motion_detected":
        diff_val = event.payload.get("difference", 0.0)
        agent_prompt = (
            f"[SYSTEM EVENT: MOTION DETECTED] Camera detected physical movement "
            f"(intensity {diff_val:.1f}). Inspect the scene to verify status."
        )

    else:
        agent_prompt = f"[SYSTEM EVENT: {event.event_type} from {event.source}]: {event.payload}"

    _queue_counter += 1
    meta = {
        "source": event.source,
        "event_type": event.event_type,
        "timestamp": time.time(),
    }
    await event_queue.put((priority, _queue_counter, agent_prompt, meta))
    log_event_audit(event.event_type, event.source, priority, "enqueued", detail="Added to priority queue")
    await broadcast_sse(
        "event_received",
        {"source": event.source, "event_type": event.event_type, "priority": priority, "prompt": agent_prompt},
    )
    await broadcast_queue_state()
    return {"status": "enqueued", "priority": priority, "event_type": event.event_type, "prompt": agent_prompt}


@app.get("/events/stream")
async def event_stream(request: Request):
    """
    Server-Sent Events (SSE) feed. Connect via phone/laptop browser or curl
    to receive live agent thoughts and actions.
    """
    queue = asyncio.Queue()
    sse_subscribers.append(queue)

    async def sse_generator():
        try:
            yield "data: {\"type\": \"connected\", \"message\": \"SSE stream active\"}\n\n"
            while not is_shutting_down:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=1.0)
                    if data is None or is_shutting_down:
                        break
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    continue
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            if queue in sse_subscribers:
                sse_subscribers.remove(queue)

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def dashboard_index():
    """Serves the interactive Portrait / Landscape web dashboard."""
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>WAVEGO Dashboard template not found</h1>"


@app.get("/video/stream")
async def video_stream(request: Request, cam: str = "tracking_cam"):
    """
    Streams live MJPEG video from the MuJoCo simulation.
    cam: 'tracking_cam' (3rd-person) or 'front_camera' (robot eye).
    Uses asyncio.to_thread to keep the event loop responsive to Ctrl+C (SIGINT).
    """
    async def frame_generator():
        loop = asyncio.get_running_loop()
        try:
            while not await request.is_disconnected():
                robot = get_robot()
                if hasattr(robot, "get_jpeg_frame"):
                    # Offload OpenGL rendering to dedicated single-worker thread to eliminate X11 BadAccess
                    render_call = functools.partial(
                        robot.get_jpeg_frame,
                        camera_name=cam,
                        width=480,
                        height=360,
                        quality=75,
                    )
                    frame_bytes = await loop.run_in_executor(render_executor, render_call)
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                await asyncio.sleep(0.04)  # ~25 FPS
        except (asyncio.CancelledError, GeneratorExit):
            pass

    return StreamingResponse(
        frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )

