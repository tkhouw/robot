"""
FastAPI Event Stream Gateway for Scout: Autonomous Pet Companion.
Exposes endpoints for external event triggers (Nest Doorbell, Vision Motion)
and voice/text commands, streaming agent reasoning via Server-Sent Events (SSE).
"""

import asyncio
import base64
import functools
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from google.genai import types as genai_types
from pydantic import BaseModel

from app.agent import QuadrupedCognitiveRuntime
from app.config import config
from app.tools import get_robot, get_action_manager

app = FastAPI(title="Scout Autonomous Pet Companion Gateway")

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
    "battery_low": 60.0,      # 1 minute debounce
    "battery_critical": 30.0, # 30 seconds debounce
    "battery_charged": 300.0, # 5 minutes debounce
    "battery_depleted": 600.0,# 10 minutes debounce
}

# Battery cycle event flags
_battery_full_notified: bool = False
_battery_depleted_notified: bool = False


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


def format_prompt_display(prompt: Any) -> str:
    """Formats prompts into clean, JSON-serializable strings for UI and SSE display."""
    if isinstance(prompt, list):
        text_parts = [p.text for p in prompt if hasattr(p, "text") and p.text]
        non_text_count = sum(1 for p in prompt if not (hasattr(p, "text") and p.text))
        combined_text = "\n".join(text_parts) if text_parts else ""
        if non_text_count > 0:
            suffix = f"\n[+{non_text_count} non-text part{'s' if non_text_count > 1 else ''} attached]"
            return combined_text + suffix if combined_text else suffix.strip()
        return combined_text
    return str(prompt)


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
                "prompt": format_prompt_display(prompt),
                "source": meta.get("source", "unknown"),
                "event_type": meta.get("event_type", "unknown"),
                "timestamp": meta.get("timestamp", time.time()),
            })
    except Exception:
        pass

    formatted_active = None
    if active_task:
        formatted_active = dict(active_task)
        if "prompt" in formatted_active:
            formatted_active["prompt"] = format_prompt_display(formatted_active["prompt"])

    state_payload = {
        "is_busy": is_agent_busy,
        "queue_size": event_queue.qsize(),
        "active_task": formatted_active,
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
    1: doorbell_press, action_completed, action_stalled (Action completions & alerts)
    2: action_progress (Normal action progress feedback)
    3: motion_detected, passive camera alert, generic sensor events
    """
    if event.event_type in ("user_message", "battery_critical", "battery_depleted"):
        return 0
    if event.event_type in (
        "doorbell_press",
        "action_completed",
        "action_stalled",
        "bifurcation_disambiguation_required",
        "battery_low",
        "battery_charged",
    ):
        return 1
    if event.event_type == "action_progress":
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
    message = json.dumps({"type": event_type, "data": data, "timestamp": time.time()}, default=str)
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

        # Format human-readable prompt string for SSE streaming
        prompt_display = prompt_text
        if isinstance(prompt_text, list):
            # Extract and join all text parts for full UI display and append non-text parts count
            text_parts = [p.text for p in prompt_text if hasattr(p, "text") and p.text]
            non_text_count = sum(1 for p in prompt_text if not (hasattr(p, "text") and p.text))
            combined_text = "\n".join(text_parts) if text_parts else ""
            if non_text_count > 0:
                suffix = f"\n[+{non_text_count} non-text part{'s' if non_text_count > 1 else ''} attached]"
                prompt_display = combined_text + suffix if combined_text else suffix.strip()
            else:
                prompt_display = combined_text

        await broadcast_sse("agent_turn_start", {"priority": priority, "prompt": prompt_display})
        await broadcast_queue_state()

        try:
            # Enforce an overall 20s timeout per stimulus turn; bounds are natively enforced by ADK max_llm_calls
            async with asyncio.timeout(20.0):
                async for adk_event in runtime.execute_prompt(prompt_text):
                    event_dict = adk_event.model_dump(mode="json", exclude_none=True)

                    # Extract tool calls and responses via ADK methods
                    if hasattr(adk_event, "get_function_calls"):
                        for fc in adk_event.get_function_calls():
                            print(f"[ADK TOOL CALL]: {fc.name}({fc.args})")

                    if hasattr(adk_event, "get_function_responses"):
                        for fr in adk_event.get_function_responses():
                            print(f"[ADK TOOL RESULT]: {fr.name} -> {fr.response}")

                    if adk_event.content and getattr(adk_event.content, "parts", None):
                        for part in adk_event.content.parts:
                            if hasattr(part, "text") and part.text:
                                print(f"[ADK AGENT THOUGHT/RESPONSE]: {part.text}")

                    await broadcast_sse("agent_adk_event", event_dict)

        except TimeoutError:
            print("[AGENT TIMEOUT] Turn exceeded timeout limit; concluding turn and releasing queue.")
            await broadcast_sse("agent_error", {"error": "Turn timed out; proceeding to next event."})
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
    """Pushes real-time telemetry over the existing SSE stream (4 Hz) and monitors system power."""
    while not is_shutting_down:
        try:
            robot = get_robot()
            bat_pct = robot.get_battery_level()
            charging = robot.is_charging()

            # Continuous Power & Battery Cycle Monitor
            global _battery_full_notified, _battery_depleted_notified
            low_thresh = getattr(config, "battery_low_threshold_pct", 20.0)
            crit_thresh = getattr(config, "battery_critical_threshold_pct", 5.0)

            # 1. 100% Full Charge Alert (wakes up agent upon completing charging)
            if bat_pct >= 100.0 and charging:
                if not _battery_full_notified:
                    _battery_full_notified = True
                    asyncio.create_task(
                        receive_event(
                            RobotStimulusEvent(
                                source="battery_monitor",
                                event_type="battery_charged",
                                payload={"battery_pct": 100.0},
                            )
                        )
                    )
            elif bat_pct < 98.0:
                _battery_full_notified = False

            # 2. 0% Depleted Shutdown Alert (fires once, halts further critical alerts)
            if bat_pct <= 0.0:
                if not _battery_depleted_notified:
                    _battery_depleted_notified = True
                    asyncio.create_task(
                        receive_event(
                            RobotStimulusEvent(
                                source="battery_monitor",
                                event_type="battery_depleted",
                                payload={"battery_pct": 0.0},
                            )
                        )
                    )
            else:
                _battery_depleted_notified = False
                # 3. Low & Critical Alerts (only fire while battery > 0% and robot is not charging)
                if not charging and bat_pct <= low_thresh:
                    event_type = "battery_critical" if bat_pct <= crit_thresh else "battery_low"
                    key = f"battery_monitor:{event_type}"
                    now_ts = time.time()
                    last_ts = _last_event_timestamps.get(key, 0.0)
                    debounce_win = DEBOUNCE_WINDOWS.get(event_type, 60.0)

                    # Only dispatch if debounce window has elapsed; receive_event records the timestamp
                    if (now_ts - last_ts) >= debounce_win:
                        asyncio.create_task(
                            receive_event(
                                RobotStimulusEvent(
                                    source="battery_monitor",
                                    event_type=event_type,
                                    payload={"battery_pct": bat_pct},
                                )
                            )
                        )

            if sse_subscribers:
                pos, _ = robot.get_pose()
                rpy = robot.get_imu()
                action_mgr = get_action_manager()
                motion_sensor = getattr(app.state, "motion_sensor", None)

                lidar_info = None
                action_details = action_mgr.action_details or {}
                if "lidar" in action_details:
                    lidar_info = action_details["lidar"]
                elif hasattr(robot, "get_lidar_scan"):
                    scan = robot.get_lidar_scan(num_samples=72)
                    if scan:
                        lidar_info = {
                            "min_distance_m": scan.get("min_distance_m"),
                            "sectors": scan.get("sectors"),
                            "is_detouring": False,
                        }

                data = {
                    "pos": [round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)],
                    "yaw": round(float(rpy[2]), 1),
                    "battery": {
                        "pct": round(float(bat_pct), 1),
                        "is_charging": bool(charging),
                    },
                    "is_busy": is_agent_busy or action_mgr.is_busy,
                    "active_action": action_mgr.active_action,
                    "action_progress": action_mgr.progress_pct,
                    "action_details": action_details,
                    "lidar": lidar_info,
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

    async def auto_wake_exploration():
        # Wait until user opens the dashboard (SSE stream connected) so they see the wake-up action
        while not sse_subscribers and not is_shutting_down:
            await asyncio.sleep(0.5)

        if is_shutting_down:
            return

        # Brief settling delay for browser DOM and SVG elements to mount
        await asyncio.sleep(1.5)
        wake_message = (
            "Good morning! Wake up and begin your playful domestic patrol. "
            "Say hello, explore the house, and check out what's going on around the home."
        )
        await receive_event(
            RobotStimulusEvent(
                source="domestic_owner",
                event_type="user_message",
                payload={"message": wake_message},
            )
        )

    asyncio.create_task(auto_wake_exploration())


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
    # Protect priority 0 (user) and priority 1 (doorbell, action_completed, action_stalled) from being dropped!
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

    elif event.event_type == "action_progress":
        # Live action progress broadcast directly to dashboard UI widget (does not interrupt agent)
        await broadcast_sse("action_progress", event.payload)
        return {"status": "broadcasted", "event_type": "action_progress"}

    elif event.event_type == "action_completed":
        action_name = event.payload.get("action", "action")
        result = event.payload.get("result", "completed successfully")
        pos = event.payload.get("position", [0, 0])
        agent_prompt = (
            f"[SYSTEM EVENT: ACTION COMPLETED] Action '{action_name}' completed. "
            f"Details: {result}. Current position: {pos}. "
            f"Conclude your next step or report status."
        )

    elif event.event_type == "action_stalled":
        action_name = event.payload.get("action", "action")
        pos = event.payload.get("current_pos", [0, 0])
        agent_prompt = (
            f"[SYSTEM ALERT: ACTION STALLED] Action '{action_name}' was blocked by a physical obstruction "
            f"at position {pos}! Locomotion stopped. Please call `capture_camera_image` to inspect, "
            f"and navigate around the blockage."
        )

    elif event.event_type == "bifurcation_disambiguation_required":
        landmark_name = event.payload.get("landmark", "target landmark")
        score_l = event.payload.get("score_left", 0.5)
        score_r = event.payload.get("score_right", 0.5)
        b64_l = event.payload.get("image_left_b64", "")
        b64_r = event.payload.get("image_right_b64", "")

        instructions_text = (
            f"[SYSTEM QUERY: BIFURCATION AMBIGUITY] While navigating towards '{landmark_name}', an obstacle directly ahead "
            f"presents two comparable detour options (LiDAR obstacle clearance: Left={score_l:.2f}, Right={score_r:.2f}). "
            f"The robot has paused and instantaneously captured directional camera views via its front ultra-wide camera lens:\n"
            f"- Image 1: Left corridor view\n"
            f"- Image 2: Right corridor view\n\n"
            f"Examine both images carefully. Assign two preference scores between 0.0 (strongly obstructed / blocked) "
            f"and 1.0 (clear, open, highly traversable route) indicating your preference for the left and right paths "
            f"based on perceived traversability and your understanding of the smart home layout. "
            f"Call `set_corridor_preference(score_left, score_right)` with your scores."
        )

        parts: List[genai_types.Part] = [genai_types.Part.from_text(text=instructions_text)]
        if b64_l:
            try:
                img_l_bytes = base64.b64decode(b64_l)
                parts.append(genai_types.Part.from_bytes(data=img_l_bytes, mime_type="image/jpeg"))
            except Exception as e:
                print(f"[GATEWAY] Failed decoding left snapshot: {e}")
        if b64_r:
            try:
                img_r_bytes = base64.b64decode(b64_r)
                parts.append(genai_types.Part.from_bytes(data=img_r_bytes, mime_type="image/jpeg"))
            except Exception as e:
                print(f"[GATEWAY] Failed decoding right snapshot: {e}")

        agent_prompt = parts

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

    elif event.event_type in ("battery_low", "battery_critical"):
        pct = event.payload.get("battery_pct", 0.0)
        is_crit = event.event_type == "battery_critical"
        severity = "CRITICAL" if is_crit else "LOW"
        agent_prompt = (
            f"[SYSTEM ALERT: BATTERY {severity}] Current battery level is {pct:.1f}%. "
            f"Power is running low. Please navigate to your charging dock immediately using `return_to_charging_dock()` to recharge."
        )

    elif event.event_type == "battery_charged":
        agent_prompt = (
            "[SYSTEM EVENT: BATTERY FULLY CHARGED] Battery has reached 100% capacity! "
            "You are fully charged and ready for instructions, room patrol, or exploration."
        )

    elif event.event_type == "battery_depleted":
        agent_prompt = (
            "[SYSTEM EVENT: BATTERY DEPLETED] Battery has reached 0.0%! "
            "All locomotion has been shut down and you are unable to move until physically placed on your charging dock."
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

    # Serialize prompt safely for SSE broadcast and HTTP response if it contains Part objects
    if isinstance(agent_prompt, list):
        text_pieces = [p.text for p in agent_prompt if hasattr(p, "text") and p.text]
        non_text_count = sum(1 for p in agent_prompt if not (hasattr(p, "text") and p.text))
        combined_text = "\n".join(text_pieces) if text_pieces else ""
        if non_text_count > 0:
            suffix = f"\n[+{non_text_count} non-text part{'s' if non_text_count > 1 else ''} attached]"
            prompt_summary = combined_text + suffix if combined_text else suffix.strip()
        else:
            prompt_summary = combined_text
    else:
        prompt_summary = str(agent_prompt)

    await broadcast_sse(
        "event_received",
        {"source": event.source, "event_type": event.event_type, "priority": priority, "prompt": prompt_summary},
    )
    await broadcast_queue_state()
    return {"status": "enqueued", "priority": priority, "event_type": event.event_type, "prompt": prompt_summary}


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


@app.get("/landmarks")
async def get_landmarks():
    """Returns the registered landmarks and their coordinates from config.py."""
    return {"landmarks": config.landmarks}


@app.get("/", response_class=HTMLResponse)
async def dashboard_index():
    """Serves the interactive Portrait / Landscape web dashboard."""
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Scout Dashboard template not found</h1>"


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
                    # Front camera uses widescreen 640x320 (2:1 aspect ratio) matched to ultra-wide lens
                    # 3rd-person spectator camera uses standard 480x360 (4:3 aspect ratio)
                    s_width = 640 if cam == "front_camera" else 480
                    s_height = 320 if cam == "front_camera" else 360

                    # Offload OpenGL rendering to dedicated single-worker thread to eliminate X11 BadAccess
                    render_call = functools.partial(
                        robot.get_jpeg_frame,
                        camera_name=cam,
                        width=s_width,
                        height=s_height,
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

