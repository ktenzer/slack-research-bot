from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List
import inspect, dataclasses
import random, time
from google.api_core.exceptions import TooManyRequests

from langgraph.graph import StateGraph
from vertexai.generative_models import (
    Candidate,
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Part,
)

# -- helper imported from your existing codebase
from ..agent.tool import create_enhanced_tool


@dataclass
class GraphState:
    """Mutable state passed between LangGraph nodes."""
    contents: List[Content] = field(default_factory=list)
    tool_calls: List | None = field(default_factory=list)  # Vertex function calls


# --------------------------------------------------------------------------- #
#  Graph builder
# --------------------------------------------------------------------------- #

def build_conversation_graph(*, model, vertex_tool, tool_map):
    graph = StateGraph(dict)                       # <- state is just a dict

    # ------------------------------------------------------------------ #
    # LLM node
    # ------------------------------------------------------------------ #
    MAX_RETRIES = 5

    def llm_node(state: dict) -> dict:
        contents = state["contents"]

        for attempt in range(MAX_RETRIES):
            try:
                raw_rsp = model.generate_content(
                    contents=[c.to_dict() for c in contents],
                    generation_config=GenerationConfig(temperature=0),
                    tools=[vertex_tool],
                ).to_dict()
                break                                # success -> exit retry loop
            except TooManyRequests as err:
                sleep_s = (2 ** attempt) + random.uniform(0, 1)
                print(f"429 received, retrying in {sleep_s:.1f}s ({attempt+1}/{MAX_RETRIES})")
                time.sleep(sleep_s)
        else:
            # after MAX_RETRIES attempts
            raise

        candidate = GenerationResponse.from_dict(raw_rsp).candidates[0]
        contents.append(candidate.content)
        state["tool_calls"] = candidate.function_calls or []
        return state
    # ------------------------------------------------------------------ #
    # Tool-execution node
    # ------------------------------------------------------------------ #
    def tool_node(state: dict) -> dict:
        parts: List[Part] = []

        for fc in state["tool_calls"]:
            fn_name = fc.name
            fn_args = next(iter(fc.args.values()), {})        # JSON dict
            fn = tool_map[fn_name]

            # --- NEW: rebuild the proper argument object if needed -------------
            sig = inspect.signature(fn)
            first_param = next(iter(sig.parameters.values()))
            param_type = first_param.annotation

            if dataclasses.is_dataclass(param_type):          # e.g. SlackChannelRequest
                arg_obj = param_type(**fn_args)               # → dataclass instance
                result = fn(arg_obj)
            else:
                result = fn(fn_args)                          # plain dict or primitives
            # -------------------------------------------------------------------

            parts.append(
                Part.from_function_response(
                    name=fn_name,
                    response={"content": result},
                )
            )

        state["contents"].append(Content(role="user", parts=parts))
        state["tool_calls"] = []
        return state

    graph.add_node("llm", llm_node)
    graph.add_node("tool", tool_node)

    graph.set_entry_point("llm")
    graph.add_conditional_edges(
        "llm", lambda s: "tool" if s["tool_calls"] else "__end__"
    )
    graph.add_edge("tool", "llm")

    return graph.compile()