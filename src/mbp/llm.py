import json
import functools
from .core import *

__all__ = ["build_icl_inputs", "load_icl_inputs"]


def encode_message(context):
    return json.dumps(context, ensure_ascii=False, separators=(",", ":"))


FORMAT_HINT = "Expecting format of => instruction: str, format: dict, examples: [dict]"


def build_icl_inputs(instruction, format={}, examples=[]):
    assert isinstance(instruction, str) and isinstance(format, dict) and len(format) > 0 and isinstance(examples, list), FORMAT_HINT
    format = {k: {"type": v} if isinstance(v, str) else v for k, v in format.items()}
    res = [instruction]
    if examples:
        res.append("Example:" if len(examples) == 1 else "Examples:")
        inp, out = {}, {}
        for e in examples:
            for k, v in e.items():
                if k in format:
                    out[k] = v
                else:
                    inp[k] = v
            res.append(f"user: {encode_message(inp)}\nassistant: {encode_message(out)}")
    return "\n\n".join(res), format


@functools.lru_cache(maxsize=None)
def load_icl_inputs(path):
    return build_icl_inputs(**load_yaml(path))
