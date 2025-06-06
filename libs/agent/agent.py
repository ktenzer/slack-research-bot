import asyncio
import logging
import os
import uuid
from typing import Callable, Dict, List, Optional

import vertexai
from vertexai.generative_models import (
    Content,
    GenerativeModel,
    Part,
)

from ..agent.tool import create_enhanced_tool                     # ← unchanged
from .workflow import GraphState, build_conversation_graph  # ← new LangGraph workflow


class Agent:
    """
    Same public surface as before (async context-manager, prompt(), thoughts()) but
    executed entirely in-process with LangChain/LangGraph.
    """

    def __init__(
        self,
        *,
        gcp_project: Optional[str] = None,
        region: str = "us-central1",
        model_name: str = "gemini-2.0-flash",
        instruction: str = (
            "You are a store-support API assistant to help with online orders."
        ),
        functions: List[Callable] | None = None,
    ) -> None:
        self.gcp_project = gcp_project or os.getenv("GCP_PROJECT_ID")
        self.region = region
        self.model_name = model_name
        self.instruction = instruction
        self.functions: List[Callable] = functions or []

        # Vertex initialisation
        vertexai.init(project=self.gcp_project, location=self.region)
        self._model = GenerativeModel(
            self.model_name,
            system_instruction=[self.instruction],
        )

        # Build Vertex-compatible tool schema once
        self._vertex_tool = create_enhanced_tool(self.functions)

        # Map function name ➜ python callable for the tool node
        self._tool_map: Dict[str, Callable] = {fn.__name__: fn for fn in self.functions}

        # Persistent conversation state
        self._contents: List[Content] = []

        # LangGraph executor (compiled graph)
        self._graph = build_conversation_graph(
            model=self._model,
            vertex_tool=self._vertex_tool,
            tool_map=self._tool_map,
        )

        self._terminated = False
        logging.basicConfig(level=logging.INFO)

    # --------------------------------------------------------------------- #
    #  Async context-manager helpers
    # --------------------------------------------------------------------- #
    async def __aenter__(self) -> "Agent":
        # Kick off an empty workflow instance (mirrors Temporal start_workflow)
        self._workflow_id = str(uuid.uuid4())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            await self.prompt("END")
        finally:
            self._terminated = True

    # --------------------------------------------------------------------- #
    #  Public API: prompt / thoughts
    # --------------------------------------------------------------------- #
    async def prompt(self, prompt: str) -> str:
        if self._terminated:
            raise RuntimeError("Agent has been terminated.")

        if prompt == "END":
            self._terminated = True
            return ""

        # 1) push user message into history
        self._contents.append(Content(role="user", parts=[Part.from_text(prompt)]))

        # 2) build the incoming state **as a plain dict**
        state_in = {"contents": list(self._contents), "tool_calls": []}

        # 3) run the graph – result is an AddableValuesDict
        state_out = self._graph.invoke(state_in)

        # 4) pull the updated history back out
        self._contents = state_out["contents"]

        # 5) return the last model text
        for content in reversed(self._contents):
            if content.role == "model":
                for part in content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        return text
        return ""

async def thoughts(self, watermark: int) -> List[str]:
    texts, idx = [], 0
    for content in self._contents:
        if content.role == "model":
            for part in content.parts:
                if getattr(part, "text", None):
                    if idx > watermark:          #  ✅ strict “greater than”
                        texts.append(part.text)
                    idx += 1
    return texts