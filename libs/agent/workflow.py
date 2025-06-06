from __future__ import annotations

import inspect
import json
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Union

import openai                             
from langchain_openai import ChatOpenAI
from langchain.schema.messages import AIMessage
from langgraph.graph import StateGraph

@dataclass
class GraphState:
    messages: List[Dict] = field(default_factory=list)   # OpenAI-style chat msgs
    tool_calls: List[Dict] | None = field(default_factory=list)

def _parse_call(call: Dict, idx: int) -> tuple[str, str, dict]:
    """
    Return (tool_call_id, fn_name, fn_args) for both old and new OpenAI schemas.
    """
    if "name" in call:                                    # legacy flat format
        fn_name = call["name"]
        raw_args = call.get("arguments", "{}")
        tool_call_id = call.get("id") or f"legacy_{idx}_{fn_name}"
    else:                                                 # new nested format
        fn_name = call["function"]["name"]
        raw_args = call["function"].get("arguments", "{}")
        tool_call_id = call["id"]
    fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    return tool_call_id, fn_name, fn_args


def build_conversation_graph(
    *,
    model: ChatOpenAI,
    functions_schema: List[Dict],
    tool_map: Dict[str, Callable],
    ):
    """
    LangGraph with two nodes:
      • llm_node  call the LLM (with OpenAI tool calling)
      • tool_node execute python tools requested by the LLM
    """


    graph = StateGraph(dict)

    MAX_RETRIES = 5

    # LLM node 
    def llm_node(state: Dict) -> Dict:
        msgs = state["messages"]

        for attempt in range(MAX_RETRIES):
            try:
                openai_tools = [
                    {"type": "function", "function": f}  
                    for f in functions_schema             
                ]
                resp: AIMessage = model.invoke(
                    msgs,
                    tools=openai_tools,
                    temperature=0,
                )
                break
            except openai.RateLimitError:
                backoff = (2 ** attempt) + random.random()
                time.sleep(backoff)
        else:
            raise

        resp_dict = {
            "role": "assistant",
            "content": resp.content,        
            **resp.additional_kwargs,          
        }

        msgs.append(resp_dict)
        state["tool_calls"] = resp_dict.get("tool_calls", [])
        return state

    # Tool node 
    def tool_node(state: Dict) -> Dict:
        msgs = state["messages"]
        results: List[Dict] = []

        for idx, call in enumerate(state["tool_calls"]):
            tool_call_id, fn_name, fn_args = _parse_call(call, idx)

            if fn_name not in tool_map:
                print(f"[tool_node] unknown tool {fn_name}, skipping")
                continue

            py_fn = tool_map[fn_name]

            sig = inspect.signature(py_fn)
            first_param = next(iter(sig.parameters.values()))
            param_name = first_param.name
            param_type = first_param.annotation

            if inspect.isclass(param_type) and hasattr(param_type, "__dataclass_fields__"):
                # If LLM wrapped args like {"request": {...}}, unwrap first
                if len(fn_args) == 1 and param_name in fn_args:
                    fn_args = fn_args[param_name]
                out = py_fn(param_type(**fn_args))
            else:
                out = py_fn(fn_args)

            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,     
                    "name": fn_name,                     
                    "content": json.dumps({"content": out}),
                }
            )

        msgs.extend(results)
        state["tool_calls"] = []          
        return state

    # Graph nodes
    graph.add_node("llm", llm_node)
    graph.add_node("tool", tool_node)

    graph.set_entry_point("llm")
    graph.add_conditional_edges(
        "llm", lambda s: "tool" if s["tool_calls"] else "__end__"
    )
    graph.add_edge("tool", "llm")

    return graph.compile()