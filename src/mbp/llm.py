import json
import functools
from .core import *

__all__ = ["encode_content", "build_system_message", "load_system_message", "add_to_messages", "build_messages"]


def encode_content(context):
    return json.dumps(context, ensure_ascii=False)


FORMAT_HINT = "Expecting format of => instruction: str, format: dict, examples: [dict]"


def build_system_message(instruction, format={}, examples=[]):
    assert isinstance(instruction, str) and isinstance(format, dict) and len(format) > 0 and isinstance(examples, list), FORMAT_HINT
    format = {k: {"type": v} if isinstance(v, str) else v for k, v in format.items()}
    res = [instruction]
    if examples:
        res.append("Example:" if len(examples) == 1 else "Examples:")
        c, o = {}, {}
        for e in examples:
            for k, v in e.items():
                if k in format:
                    o[k] = v
                else:
                    c[k] = v
            res.append(f"user: {encode_content(c)}\nresp: {encode_content(o)}")
    return "\n\n".join(res), format


@functools.lru_cache(maxsize=None)
def load_system_message(path):
    return build_system_message(**load_yaml(path))


def add_to_messages(messages, role, content):
    if isinstance(content, dict):
        content = encode_content(content)
    messages.append({"role": role, "content": content})


def build_messages(system_message, context):
    res = []
    add_to_messages(res, "system", system_message)
    add_to_messages(res, "user", context)
    return res
