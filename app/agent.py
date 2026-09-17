import os
from typing import Any, AsyncGenerator, Optional
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
    stop_action,
    return_to_charging_dock,
    perform_handshake,
    capture_camera_image,
    announce_message,
    read_telemetry,
    read_lidar_scan,
)

SYSTEM_INSTRUCTION = """You are a small autonomous cognitive domestic quadruped robot agent.  You function as a friendly pet to your owner.
You have physical motor tools to navigate your environment, inspect scenes with your camera, scan obstacles with your 360 DTOF LiDAR, announce speech, and charge your battery.

### Environment & Landmarks:
You operate in a smart home environment with known landmarks:
- 'charging_dock': Your home base with metal contact plates (0.0, 0.0).
- 'front_door': The entrance of the house (4.4, 0.0).
- 'living_room': Central relaxation area (1.0, 1.2).
- 'kitchen': Cooking and food preparation area (3.5, -1.2).

### Guidelines & Operational Rules:
1. Free Play
   - When you are not explicitly instructed to do anything else, your job is to playfully explore your environment.  Walk around, speak to yourself, and use your camera and LiDAR.
   - If you see a human, assume it is your owner and follow them around.  You enjoy their company just like a loyal pet likes their owner.  If you don't see them, you can ask where they are.  If they hear you, they may answer you.
2. Speaking
   - No one will be able to hear any of your thoughts unless you speak verbally using your announce_message tool.
3. When you receive a [DOORBELL RING] or visitor alert:
   - Call `navigate_to_landmark('front_door')`.
   - Conclude your turn: "Heading to the front door to greet the visitor."
   - When you subsequently receive [ACTION COMPLETED] for the front door, use `capture_camera_image` to inspect the entrance and `announce_message` to politely greet the visitor.
   - Do NOT perform a handshake unless explicitly requested by the user.
4. Stopping & Interruption:
   - If the user says 'Stop!', 'Halt!', 'Wait!', or tells you to stop, IMMEDIATELY call `stop_action()`.
5. When instructed to dock, charge, or enter idle sleep:
   - Call `return_to_charging_dock()`, and conclude your turn.
5. Multimodal Vision & LiDAR Sensing:
   - You have access to `capture_camera_image` to visually inspect scenes and identify visitors or blockages.
   - You have access to `read_lidar_scan` to measure 360-degree obstacle clearances in 8 directional sectors.
6. Movement:
   - For simple exploratory movements, use `walk_forward`, `walk_backward`, `turn_left`, and `turn_right`.
   - For traveling to known landmarks, use `navigate_to_landmark`.
7. Obstacle & Stall Recovery:
   - If you receive an [ACTION STALLED] alert (e.g. detour timeout or blocked path):
     - Check `read_lidar_scan` to understand where the obstacle is located.
     - Call `capture_camera_image` to visually inspect what is blocking you.  Note: just because you see something in the image doesn't mean it is blocking you.  You need to reason about how much space is between you and whatever you see through your camera.
     - Manually navigate away from the obstacle using your "walk" and "turn" tools.  Iteratively check your LiDAR and camera as needed for guidance.
     - Once cleared, resume normal exploration or navigation towards your target landmark.
"""


class MultimodalToolOpenAILlm(OpenAILlm):
    """
    Subclasses ADK OpenAILlm to preserve multimodal media parts (e.g. image bytes)
    returned by tools inside role: 'tool' messages.
    """

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[Any, None]:
        import base64
        import json
        from google.adk.labs.openai._openai_llm import _content_to_openai_messages

        messages = []
        if llm_request.config and llm_request.config.system_instruction:
            messages.append({
                "role": "system",
                "content": llm_request.config.system_instruction,
            })

        for content in llm_request.contents or []:
            converted = _content_to_openai_messages(content)
            # Inspect if content contains a function_response with multimodal parts
            for part in getattr(content, "parts", []) or []:
                if getattr(part, "function_response", None) and getattr(part.function_response, "parts", None):
                    resp_id = part.function_response.id or ""
                    for msg in converted:
                        if msg.get("role") == "tool" and msg.get("tool_call_id") == resp_id:
                            text_body = msg.get("content", "")
                            content_list = [{"type": "text", "text": text_body}]
                            for subpart in part.function_response.parts:
                                if getattr(subpart, "inline_data", None) and subpart.inline_data.data:
                                    mime = subpart.inline_data.mime_type or "image/png"
                                    b64 = base64.b64encode(subpart.inline_data.data).decode("utf-8")
                                    content_list.append({
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                                    })
                            msg["content"] = content_list
            messages.extend(converted)

        # Delegate kwargs assembly and LLM execution using original methods
        from google.adk.labs.openai._openai_llm import (
            _function_declaration_to_openai_tool,
            _response_to_llm_response,
            enforce_strict_openai_schema,
        )
        from pydantic import BaseModel
        import copy

        tools = []
        if (
            llm_request.config
            and llm_request.config.tools
            and llm_request.config.tools[0].function_declarations
        ):
            tools = [
                _function_declaration_to_openai_tool(tool)
                for tool in llm_request.config.tools[0].function_declarations
            ]

        tool_choice = "auto" if tools else None

        response_format = None
        if llm_request.config and llm_request.config.response_schema:
            schema = llm_request.config.response_schema
            schema_name = "response"
            schema_dict = {}

            if isinstance(schema, type) and issubclass(schema, BaseModel):
                schema_dict = schema.model_json_schema()
                schema_name = schema.__name__
            elif isinstance(schema, BaseModel):
                schema_dict = schema.__class__.model_json_schema()
                schema_name = schema.__class__.__name__
            elif isinstance(schema, dict):
                schema_dict = copy.deepcopy(schema)
                if "title" in schema_dict:
                    schema_name = str(schema_dict["title"])

            if schema_dict:
                enforce_strict_openai_schema(schema_dict)
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema_dict,
                    },
                }
        elif (
            llm_request.config
            and llm_request.config.response_mime_type == "application/json"
        ):
            response_format = {"type": "json_object"}

        kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": tools if tools else None,
            "tool_choice": tool_choice,
            "max_tokens": self.max_tokens,
            "response_format": response_format,
        }

        if llm_request.config:
            if getattr(llm_request.config, "temperature", None) is not None:
                kwargs["temperature"] = llm_request.config.temperature
            if getattr(llm_request.config, "top_p", None) is not None:
                kwargs["top_p"] = llm_request.config.top_p
            if getattr(llm_request.config, "stop_sequences", None):
                kwargs["stop"] = llm_request.config.stop_sequences
            if getattr(llm_request.config, "max_output_tokens", None) is not None:
                kwargs["max_tokens"] = llm_request.config.max_output_tokens

        if not stream:
            response = await self._openai_client.chat.completions.create(**kwargs)
            yield _response_to_llm_response(response)
        else:
            async for response in self._generate_content_streaming(kwargs):
                yield response


def create_agent(model_name: Optional[str] = None) -> Agent:
    """Instantiates the Google ADK Agent backed by OpenRouter with multimodal tool support."""
    chosen_model = model_name or config.model_name

    client = openai.AsyncOpenAI(
        base_url=config.openrouter_base_url,
        api_key=config.openrouter_api_key,
    )

    llm = MultimodalToolOpenAILlm(
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
            stop_action,
            return_to_charging_dock,
            perform_handshake,
            capture_camera_image,
            announce_message,
            read_telemetry,
            read_lidar_scan,
        ],
    )
    return agent


class QuadrupedCognitiveRuntime:
    """
    Manages the ADK Agent, session persistence, and asynchronous turn dispatching.
    """

    def __init__(self, agent: Optional[Agent] = None):
        self.app_name = "wavego_quadruped_app"
        self.agent = agent or create_agent()
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            app_name=self.app_name,
            agent=self.agent,
            session_service=self.session_service,
        )
        self.user_id = "wavego_user"
        self.session_id = "default_wavego_session"

    async def init_session(self):
        """Initializes the session in the session service if it does not exist."""
        sess = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
        )
        if not sess:
            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=self.session_id,
            )

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

