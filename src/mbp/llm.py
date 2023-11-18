import json
import functools
from .core import *

__all__ = ['add_to_messages', 'encode_context', 'build_system_message', 'build_system_message_from_yaml', "build_messages"]


def add_to_messages(messages, role, content):
    if isinstance(content, dict):
        content = encode_context(content)
    messages.append({"role": role, "content": content})


def encode_context(context, multiline=False):
    assert isinstance(context, dict), "context must be a dict"
    res = ""
    for k, v in context.items():
        v_res = str(v)
        if "\n" in v_res or multiline:
            res += f"<{k}>\n{v_res}\n</{k}>\n"
        else:
            res += f"<{k}> {v_res}\n"
    return res


FORMAT_HINT = "expecting foramt of ... instruction: str, outputs: [str], examples (optional): [dict]"


def build_system_message(instruction, outputs=[], examples=[]):
    assert isinstance(instruction, str) and isinstance(outputs, list) \
        and len(outputs) > 0 and isinstance(outputs[0], str) and isinstance(examples, list), FORMAT_HINT

    res = []
    res.append("Your response must be in json format")
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
            res.append(f"{encode_context(c)}\n{json.dumps(o, ensure_ascii=False)}")
    return "\n\n".join(res)


@functools.lru_cache(maxsize=256)
def build_system_message_from_yaml(path):
    return build_system_message(**load_yaml(path))


def build_messages(system_message, context):
    res = []
    add_to_messages(res, "system", system_message)
    add_to_messages(res, "user", context)
    return res
