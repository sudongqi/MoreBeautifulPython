import json
import functools
from .core import *

__all__ = ["add_to_messages", "encode_context", "build_system_message", "build_system_message_from_yaml", "build_messages"]


def add_to_messages(messages, role, content):
    if isinstance(content, dict):
        content = encode_context(content)
    messages.append({"role": role, "content": content})


def encode_context(context):
    return json.dumps(context, ensure_ascii=False)


FORMAT_HINT = """Expecting format of
- instruction: str,
- outputs: [str], 
- examples: [dict]"""


def build_system_message(instruction, outputs=[], examples=[]):
    assert (
        isinstance(instruction, str)
        and isinstance(outputs, list)
        and len(outputs) > 0
        and isinstance(outputs[0], str)
        and isinstance(examples, list)
    ), FORMAT_HINT
    res = []
    _allow = ", ".join([f'"{k}"' for k in outputs])
    _key = "key" if len(outputs) == 1 else "keys"
    res.append(f"Your response must be in json format and can only allow {_allow} as {_key}")
    res.append(instruction)
    if examples:
        res.append("Here is an example:" if len(examples) == 1 else "Here are a few examples:")
        c, o = {}, {}
        for e in examples:
            for k, v in e.items():
                if k in outputs:
                    o[k] = v
                else:
                    c[k] = v
            res.append(f"{encode_context(c)}\n{encode_context(o)}")
    return "\n\n".join(res)


@functools.lru_cache(maxsize=None)
def build_system_message_from_yaml(path):
    return build_system_message(**load_yaml(path))


def build_messages(system_message, context):
    res = []
    add_to_messages(res, "system", system_message)
    add_to_messages(res, "user", context)
    return res
