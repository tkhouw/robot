import os
from typing import AsyncGenerator, Optional
import openai
import google.genai.types as types
from google.adk.agents import Agent
from google.adk.events import Event as AdkEvent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.labs.openai import OpenAILlm

from app.config import config
from app.tools import (
    walk_forward,
    walk_backward,
    turn_left,
    turn_right,
    navigate_to_landmark,
    return_to_charging_dock,
    perform_handshake,
    capture_camera_image,
    announce_message,
    read_telemetry,
)

SYSTEM_INSTRUCTION = """You are an autonomous cognitive domestic quadruped robot agent (WAVEGO Pro).
You have physical motor tools to navigate your environment, inspect scenes with your camera, announce speech, and charge your battery.

### Environment & Landmarks:
You operate in a smart home environment with known landmarks:
- 'charging_dock': Your home base with metal contact plates.
- 'front_door': The entrance of the house.
- 'living_room': Central relaxation area.
- 'kitchen': Cooking and food preparation area.

### Guidelines & Operational Rules:
1. When you receive a [DOORBELL RING] or visitor alert:
   - Navigate directly to the 'front_door'.
   - Use `capture_camera_image` to inspect the entrance.
   - Use `announce_message` to politely greet the visitor (e.g. "Hello! Someone will be with you shortly.").
   - Do NOT perform a handshake unless explicitly requested by the user.
2. When instructed to dock, charge, or enter idle sleep:
   - Call `return_to_charging_dock()`, which navigates to the dock and crouches onto the metal charging contacts.
3. Multimodal Vision Verification:
   - You have access to `capture_camera_image`. Use it when you need to inspect an area or visually verify that your objective has been reached.
4. Movement:
   - For simple movements, prefer `walk_forward`, `walk_backward`, `turn_left`, and `turn_right`.
   - For long-distance destination travel, use `navigate_to_landmark`.
5. Autonomy:
   - Think step-by-step. Call tools dynamically as needed to fulfill the request.
"""


def create_agent(model_name: Optional[str] = None) -> Agent:
    """Instantiates the Google ADK Agent backed by OpenRouter."""
    chosen_model = model_name or config.model_name

    client = openai.AsyncOpenAI(
        base_url=config.openrouter_base_url,
        api_key=config.openrouter_api_key,
    )

    llm = OpenAILlm(
        model=chosen_model,
        client=client,
    )

    agent = Agent(
        name="wavego_quadruped_agent",
        model=llm,
        instruction=SYSTEM_INSTRUCTION,
        tools=[
            walk_forward,
            walk_backward,
            turn_left,
            turn_right,
            navigate_to_landmark,
            return_to_charging_dock,
            perform_handshake,
            capture_camera_image,
            announce_message,
            read_telemetry,
        ],
    )
    return agent


class QuadrupedCognitiveRuntime:
    """
    Manages the ADK Agent, session persistence, and asynchronous turn dispatching.
    """

    def __init__(self, agent: Optional[Agent] = None):
        self.agent = agent or create_agent()
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            app_name="wavego_quadruped_app",
            agent=self.agent,
            session_service=self.session_service,
        )
        self.user_id = "wavego_user"
        self.session_id = "default_wavego_session"

    async def init_session(self):
        """Initializes the session in the session service."""
        try:
            await self.session_service.get(session_id=self.session_id)
        except Exception:
            await self.session_service.create(session_id=self.session_id)

    async def execute_prompt(self, user_prompt: str) -> AsyncGenerator[AdkEvent, None]:
        """
        Executes a turn in the ADK agent, yielding native Google ADK Event objects
        containing structured actions, tool calls, and model outputs.
        """
        await self.init_session()
        new_msg = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_prompt)],
        )
        async for event in self.runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=new_msg,
        ):
            yield event

