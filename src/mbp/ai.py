import json
import functools
from .core import *

__all__ = [
    'add_to_messages', 'encode_context', 'build_instruction', 'build_instruction_from_yaml'
]


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


def build_instruction(instruction, examples=[]):
    res = []
    res.append("Your response must be in json format")
    res.append(instruction)
    if examples:
        assert len(examples) % 2 == 0, "examples must be a list of [context, response, context, response, ...]"
        res.append("Here is an example:" if len(examples) == 2 else "Here are a few examples:")
        for i in range(0, len(examples), 2):
            res.append(f"{encode_context(examples[i])}\n{json.dumps(examples[i+1], ensure_ascii=False)}")
    return "\n\n".join(res)


@functools.lru_cache(maxsize=256)
def build_instruction_from_yaml(path):
    return build_instruction(**load_yaml(path))
