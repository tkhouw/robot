"""
FastAPI Event Stream Gateway for the WAVEGO Pro Quadruped.
Exposes endpoints for external event triggers (Nest Doorbell, Vision Motion)
and voice/text commands, streaming agent reasoning via Server-Sent Events (SSE).
"""

import asyncio
import json
import time
from typing import Any, Dict, List
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent import QuadrupedCognitiveRuntime

app = FastAPI(title="WAVEGO Pro Cognitive Agent Gateway")

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
event_queue: asyncio.Queue = asyncio.Queue()
sse_subscribers: List[asyncio.Queue] = []


class EventTriggerRequest(BaseModel):
    source: str  # e.g. "nest_doorbell", "motion_sensor"
    event_type: str  # e.g. "doorbell_press", "motion_detected"
    payload: Dict[str, Any] = {}


class CommandRequest(BaseModel):
    source: str = "user_input"
    prompt: str


async def broadcast_sse(event_type: str, data: Any):
    """Pushes a message to all active SSE browser connections."""
    message = json.dumps({"type": event_type, "data": data, "timestamp": time.time()})
    for q in list(sse_subscribers):
        try:
            await q.put(message)
        except Exception:
            pass


async def event_processor_loop():
    """Background consumer worker processing incoming events sequentially."""
    while True:
        prompt_text = await event_queue.get()
        await broadcast_sse("agent_turn_start", {"prompt": prompt_text})

        try:
            async for step_output in runtime.execute_prompt(prompt_text):
                await broadcast_sse("agent_step", {"output": step_output})
        except Exception as e:
            await broadcast_sse("agent_error", {"error": str(e)})

        await broadcast_sse("agent_turn_complete", {"status": "idle"})
        event_queue.task_done()


@app.on_event("startup")
async def startup_event():
    global runtime
    runtime = QuadrupedCognitiveRuntime()
    asyncio.create_task(event_processor_loop())


@app.post("/events/trigger")
async def trigger_event(req: EventTriggerRequest):
    """
    Ingests external webhook triggers (such as Nest Doorbell ring or motion detection).
    Leverages Nest's onboard facial recognition metadata (person_name) when present.
    """
    if req.source == "nest_doorbell" and req.event_type == "doorbell_press":
        person_name = req.payload.get("person_name")
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
    elif req.source == "motion_sensor":
        diff_val = req.payload.get("difference", 0.0)
        agent_prompt = (
            f"[SYSTEM EVENT: MOTION DETECTED] Camera detected physical movement "
            f"(intensity {diff_val:.1f}). Inspect the scene to verify status."
        )
    else:
        agent_prompt = f"[SYSTEM EVENT from {req.source}]: {req.event_type} - {req.payload}"

    await event_queue.put(agent_prompt)
    await broadcast_sse("event_received", {"source": req.source, "prompt": agent_prompt})
    return {"status": "enqueued", "prompt": agent_prompt}


@app.post("/events/command")
async def receive_command(req: CommandRequest):
    """
    Ingests phone or laptop user commands (voice transcripts or text).
    """
    user_prompt = f"[USER COMMAND]: {req.prompt}"
    await event_queue.put(user_prompt)
    await broadcast_sse("command_received", {"prompt": req.prompt})
    return {"status": "enqueued", "prompt": user_prompt}


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
            while True:
                if await request.is_disconnected():
                    break
                data = await queue.get()
                yield f"data: {data}\n\n"
        finally:
            if queue in sse_subscribers:
                sse_subscribers.remove(queue)

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
