"""
Convert python functions (with optional dataclass argument) into the JSON schema
required by the OpenAI function-calling API.
"""

import inspect
import json
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Union, get_args, get_origin

def _convert_type_to_schema(t: Any, field_name: str) -> Dict[str, Any]:
    if is_dataclass(t):
        return _dataclass_to_schema(t)
    if t is str:
        return {"type": "string"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is bool:
        return {"type": "boolean"}

    return {"type": "string", "description": f"{field_name} ({t})"}


def _dataclass_to_schema(cls: type) -> Dict[str, Any]:
    schema = {"type": "object", "properties": {}, "required": []}

    for f in fields(cls):
        schema["properties"][f.name] = _convert_type_to_schema(f.type, f.name)
        if f.default is inspect._empty and f.default_factory is inspect._empty:
            schema["required"].append(f.name)
    return schema

def create_function_schema(func: callable) -> Dict[str, Any]:
    sig = inspect.signature(func)
    params = [
        (name, p)
        for name, p in sig.parameters.items()
        if name != "self"
    ]
    if len(params) != 1:
        raise ValueError(f"{func.__name__} must have exactly one argument.")

    arg_name, param = params[0]
    param_type = param.annotation

    # Unwrap Optional[T]
    if get_origin(param_type) is Union:
        args = [a for a in get_args(param_type) if a is not type(None)]
        param_type = args[0] if args else param_type

    if is_dataclass(param_type):
        param_schema = _dataclass_to_schema(param_type)
    else:
        param_schema = _convert_type_to_schema(param_type, arg_name)

    return {
        "name": func.__name__,
        "description": inspect.getdoc(func) or f"{func.__name__} function",
        "parameters": {
            "type": "object",
            "properties": {arg_name: param_schema},
            "required": [arg_name],
        },
    }


def create_enhanced_tool(functions: List[callable]) -> List[Dict]:
    """Return a list of OpenAI-compatible function-schemas."""
    schemas: List[Dict] = []
    for fn in functions:
        try:
            schemas.append(create_function_schema(fn))
        except Exception as err:
            print(f"[tool-schema] skipping {fn.__name__}: {err}")
    return schemas