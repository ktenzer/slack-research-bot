import asyncio
import json
import logging
import os
import uuid
from typing import Callable, Dict, List, Optional

from langchain_openai import ChatOpenAI           # requires langchain-openai ≥0.1.6
from langchain.schema.messages import AIMessage

from ..agent.tool import create_enhanced_tool
from .workflow import build_conversation_graph


class Agent:
    """
    Same public interface (async-context manager, prompt(), thoughts()) but
    powered by ChatOpenAI against ANY OpenAI-compatible backend (Ollama, Groq…).
    """

    def __init__(
        self,
        *,
        model_name: str = "llama3.2",
        instruction: str = (
            "You are a store-support API assistant that helps with online orders."
        ),
        functions: List[Callable] | None = None,
        openai_api_base: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.instruction = instruction.strip()
        self.functions: List[Callable] = functions or []

        # ChatOpenAI instance (streaming off, temperature fixed to 0)
     
        self._model = ChatOpenAI(
            model=model_name,
            temperature=0,
            #base_url=openai_api_base or os.getenv("OPENAI_API_BASE"),
            #base_url='http://localhost:11434/v1',
            #openai_api_key ="ollama",
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        )

        # One-time JSON schemas for function-calling
        self._functions_schema = create_enhanced_tool(self.functions)

        # Map tool name → python callable
        self._tool_map: Dict[str, Callable] = {fn.__name__: fn for fn in self.functions}

        # Running chat history (OpenAI message dicts)
        self._messages: List[Dict] = [
            {"role": "system", "content": self.instruction}
        ]

        # Compile graph
        self._graph = build_conversation_graph(
            model=self._model,
            functions_schema=self._functions_schema,
            tool_map=self._tool_map,
        )

        self._terminated = False
        logging.basicConfig(level=logging.INFO)

    # context-manager 
    async def __aenter__(self) -> "Agent":
        self._session_id = str(uuid.uuid4())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            await self.prompt("END")
        finally:
            self._terminated = True

    # public API
    async def prompt(self, prompt: str) -> str:
        if self._terminated:
            raise RuntimeError("Agent has been terminated.")

        if prompt == "END":
            self._terminated = True
            return ""

        self._messages.append({"role": "user", "content": prompt})
        state_out = self._graph.invoke({"messages": self._messages, "tool_calls": []})
        self._messages = state_out["messages"]

        for msg in reversed(self._messages):
            if msg["role"] == "assistant" and isinstance(msg["content"], str):
                return msg["content"]
        return ""

    async def thoughts(self, watermark: int) -> List[str]:
        """Return assistant messages *after* the given watermark index."""
        texts, idx = [], 0
        for msg in self._messages:
            if msg["role"] == "assistant" and isinstance(msg["content"], str):
                if idx > watermark:
                    texts.append(msg["content"])
                idx += 1
        return texts