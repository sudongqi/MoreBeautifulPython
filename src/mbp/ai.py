import json
import functools
from .core import *

__all__ = ['add_to_messages', 'encode_context', 'build_instruction', 'build_instruction_from_yaml']


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


def build_instruction(instruction, outputs=[], examples=[]):
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
def build_instruction_from_yaml(path):
    return build_instruction(**load_yaml(path))
