import sys
import os
import shutil
import fnmatch
import random
import json
import time
import gzip
import bz2
import inspect
import itertools
import traceback
import argparse
import bisect
import asyncio
from io import StringIO
from collections.abc import Iterable
from datetime import datetime, timezone
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from termcolor import colored
from wcwidth import wcswidth
from yaml import safe_load, dump

# fmt: off
__all__ = [
    # replacement for logging
    "log", "logger", "block_logger", "set_logger", "get_logger", "recorder",
    # logging levels
    "NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
    # replacement for multiprocessing
    "Workers", "work", "async_work",
    # syntax sugar for common utilities
    "dmerge", "dsel", "ddrop", "drename", 
    # syntax sugar for common utilities
    "try_f", "type_of", "npath", "jpath", "run_dir", "lib_path",
    # handling data files
    "iterate", "iter_txt", "load_txt", "iter_jsonl", "load_jsonl", "load_json", "load_yaml", "save_txt", "save_json", "save_jsonl", "save_yaml",
    # handling paths
    "unwrap_file", "unwrap_dir", "file_basename", "dir_basename",
    # tools for file system
    "traverse", "this_dir", "dir_of", "build_dirs", "build_files", "build_dirs_for", "scan_path",
    # handling string
    "break_str", "shorten_str", "fill_str",
    # tools for debug
    "block", "block_timer", "error_msg", "summarize_exception", "debug",
    # tools for summarizations
    "prints", "print_iter", "print_table", "print_line",
    # tools for simple statistics
    "timer", "curr_time", "stats_of", "CPU_COUNT", "MIN", "MAX",
    # tools for environment
    "get_args", "run_with_args"
]
# fmt: on

NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL = 0, 10, 20, 30, 40, 50
CPU_COUNT = cpu_count()
MIN = float("-inf")
MAX = float("inf")


def _get_msg_level(level):
    labels = ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    return labels[bisect.bisect_left([NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL], level)]


def _open_files_for_logger(file):
    res = file if isinstance(file, list) else [file]
    for i in range(len(res)):
        f = res[i]
        if isinstance(f, str):
            build_dirs_for(f)
            res[i] = open(f, "w", encoding="utf-8")
    return res


class logger:
    def __init__(self, name="", file=sys.stdout, level=INFO, verbose=False):
        self.level = level
        self.file = _open_files_for_logger(file)
        self.name = name
        self.verbose = True if name else verbose

    def __call__(self, *data, level=INFO, file=None, end=None, flush=True, color=None):
        if self.level <= level:
            header = f"{curr_time()} {_get_msg_level(level)}{' ' + self.name if self.name != '' else ''}: "
            header_empty = len(header) * " "
            for f in self.file if file is None else _open_files_for_logger(file):
                for d in data:
                    lines = str(d).split("\n")
                    for idx, line in enumerate(lines):
                        if color is not None:
                            line = colored(line, color)
                        if self.verbose:
                            if idx == 0:
                                print(header, file=f, end="", flush=flush)
                            else:
                                print(header_empty, file=f, end="", flush=flush)
                        if idx == len(lines) - 1:
                            print(line, file=f, end=end, flush=flush)
                        else:
                            print(line, file=f, end=None, flush=flush)


LOGGER = logger()
BLOCK_LOGGER_SET = False


class block_logger:
    def __init__(self, name="", file=sys.stdout, level=INFO, verbose=False, can_overwrite=True):
        global LOGGER
        global BLOCK_LOGGER_SET
        self.logger_was_changed = False
        if not BLOCK_LOGGER_SET or not can_overwrite:
            self.original_logger = LOGGER
            LOGGER = logger(name, file, level, verbose)
            self.logger_was_changed = True
            BLOCK_LOGGER_SET = True

    def __enter__(self):
        pass

    def __exit__(self, *args):
        global LOGGER
        global BLOCK_LOGGER_SET
        if self.logger_was_changed:
            LOGGER = self.original_logger
            BLOCK_LOGGER_SET = False


def set_logger(name="", file=sys.stdout, level=INFO, verbose=False):
    global LOGGER
    LOGGER = logger(name, file, level, verbose)


def get_logger():
    global LOGGER
    return LOGGER


def log(*messages, level=INFO, file=None, end=None, flush=True, color=None):
    LOGGER(*messages, level=level, file=file, end=end, flush=flush, color=color)


class recorder:
    def __init__(self, captured_level=INFO):
        self.buffer = StringIO()
        self.logger = block_logger(file=self.buffer, level=captured_level, can_overwrite=False)

    def __enter__(self):
        self.logger.__enter__()
        return self

    def flush(self):
        res = self.buffer.getvalue()
        self.buffer.truncate(0)
        self.buffer.seek(0)
        return res

    def __exit__(self, *args):
        self.logger.__exit__(*args)


def error_msg(e, verbose=False, sep="\n"):
    return _np(traceback.format_exc().replace("\n", sep)) if verbose else repr(e)


def summarize_exception(e):
    res = {}
    res["error"] = error_msg(e, verbose=False)
    res["error_type"] = res["error"].split("(")[0]
    res["error_msg"] = res["error"][len(res["error_type"]) + 2 : -2]
    res["traceback"] = error_msg(e, verbose=True)
    return res


class Worker(Process):
    def __init__(self, f, inp, out, worker_id=None, cache_inp=None, build_inp=None, verbose=True):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.inp = inp
        self.out = out
        self.f = f
        self.cache_inp = cache_inp
        self.built_inp = build_inp
        if verbose:
            log("started worker-{}".format("?" if worker_id is None else worker_id))

    def run(self):
        if self.built_inp is not None:
            self.built_inp = {k: v[0](*v[1:]) for k, v in self.built_inp.items()}
        while True:
            task_id, kwargs = self.inp.get()
            try:
                if isinstance(kwargs, dict):
                    _kwargs = {k: v for k, v in kwargs.items()}
                    if self.cache_inp is not None:
                        _kwargs.update(self.cache_inp)
                    if self.built_inp is not None:
                        _kwargs.update(self.built_inp)
                    res = self.f(**_kwargs)
                else:
                    res = self.f(*kwargs)
                self.out.put({"worker_id": self.worker_id, "task_id": task_id, "task": kwargs, "res": res})
            except Exception as e:
                self.out.put(
                    {
                        "worker_id": self.worker_id,
                        "task_id": task_id,
                        "task": kwargs,
                        "error": error_msg(e, False),
                        "traceback": error_msg(e, True),
                    }
                )


class Workers:
    def __init__(self, f, num_workers=CPU_COUNT, cache_inp=None, build_inp=None, ignore_error=False, verbose=True):
        self.inp = Queue()
        self.out = Queue()
        self.workers = []
        self.task_id = 0
        self.verbose = verbose
        self.ignore_error = ignore_error
        self.f = f
        for i in range(num_workers):
            worker = Worker(f, self.inp, self.out, i, cache_inp, build_inp, verbose)
            worker.start()
            self.workers.append(worker)

    def _map(self, data):
        it = iter(data)
        running_task_num = 0
        try:
            while True:
                while running_task_num < len(self.workers):
                    task = next(it)
                    self.add_task(task)
                    running_task_num += 1
                yield self.get_result()
                running_task_num -= 1
        except StopIteration:
            for _ in range(running_task_num):
                yield self.get_result()

    def map(self, tasks, ordered=False):
        if ordered:
            saved = {}
            id_task_waiting_for = 0
            for d in self._map(tasks):
                saved[d["task_id"]] = d
                while id_task_waiting_for in saved:
                    yield saved[id_task_waiting_for]
                    saved.pop(id_task_waiting_for)
                    id_task_waiting_for += 1
        else:
            for d in self._map(tasks):
                yield d

    def add_task(self, inp):
        self.inp.put((self.task_id, inp))
        self.task_id += 1

    def get_result(self):
        res = self.out.get()
        if "error" in res:
            err_msg = "worker-{} failed task-{} : {}".format(res["worker_id"], res["task_id"], res["error"])
            if not self.ignore_error:
                self.terminate()
                assert False, err_msg
            if self.verbose:
                log(err_msg)
        elif self.verbose:
            log("worker-{} completed task-{}".format(res["worker_id"], res["task_id"]))
        return res

    def terminate(self):
        for w in self.workers:
            w.terminate()
        if self.verbose:
            log("terminated {} workers".format(len(self.workers)))


def work(f, tasks, num_workers=CPU_COUNT, cache_inp=None, build_inp=None, ordered=True, ignore_error=False, res_only=True, verbose=False):
    workers = Workers(f, num_workers, cache_inp, build_inp, ignore_error, verbose)
    for d in workers.map(tasks, ordered):
        yield d.get("res", None) if res_only else d
    workers.terminate()


def _count_and_show(kwargs):
    print_line(text=f"{kwargs['acc']}/{kwargs['total']}", width=60)
    if kwargs["metadata"]:
        print(prints(kwargs["metadata"], res=True)[:2000])
    prints(kwargs["res"], color="green")


async def async_work(f, tasks, concurrency=None, f_step=_count_and_show, ordered=True):
    sem = asyncio.Semaphore(concurrency) if concurrency is not None else None

    async def _f(idx, **kwargs):
        if sem is not None:
            async with sem:
                metadata = kwargs.pop("metadata", {})
                res = await f(**kwargs)
        else:
            metadata = kwargs.pop("metadata", {})
            res = await f(**kwargs)
        return {"idx": idx, "res": res, "metadata": metadata}

    res = []
    acc = 0
    total = len(tasks)
    async for co in asyncio.as_completed([_f(idx, **inp) for idx, inp in enumerate(tasks)]):
        acc += 1
        try:
            resp = await co
            if f_step is not None:
                f_step({**resp, "acc": acc, "total": total})
            res.append(resp)
        except:
            continue

    if ordered:
        res.sort(key=lambda x: x["idx"])

    return [{**r["metadata"], **r["res"]} if isinstance(r["res"], dict) else {"res": r["res"]} for r in res]


class timer:
    def __init__(self, msg="", level=INFO):
        self.start = None
        self.level = level
        self.msg = msg
        self.checked = False

    def __enter__(self):
        self.start = time.time()
        return self

    def check(self, msg="", reset=True):
        end_time = time.time()
        self.checked = True
        self.duration = (end_time - self.start) * 1000
        if reset:
            self.start = end_time
        log("{}took {:.3f} ms".format("" if msg == "" else f"{msg} ==> ", self.duration), level=self.level)
        return self.duration

    def __exit__(self, *args):
        if not self.checked:
            self.check(self.msg)


def curr_time(breakdown=False):
    res = str(datetime.now(timezone.utc))[:19]
    if breakdown:
        #      year           month          day             hour             minute           second
        return int(res[0:4]), int(res[5:7]), int(res[8:10]), int(res[11:13]), int(res[14:16]), int(res[17:19])
    return res


def iterate(data, first_n=None, sample_p=1.0, sample_seed=None, report_n=None):
    if sample_seed is not None:
        random.seed(sample_seed)
    if first_n is not None:
        assert first_n >= 1, "first_n should be >= 1"
    counter = 0
    total = len(data) if hasattr(data, "__len__") else "?"
    prev_time = time.time()
    for d in itertools.islice(data, 0, first_n):
        if random.random() <= sample_p:
            counter += 1
            yield d
            if report_n is not None and counter % report_n == 0:
                current_time = time.time()
                speed = report_n / (current_time - prev_time) if current_time - prev_time != 0 else "inf"
                log("{}/{} ==> {:.3f} items/s".format(counter, total, speed))
                prev_time = current_time


def _open(path, encoding="utf-8", compression=None):
    if compression is None:
        return open(path, "r", encoding=encoding)
    elif compression == "gz":
        return gzip.open(path, "rt", encoding=encoding)
    elif compression == "bz2":
        return bz2.open(path, "rb")
    else:
        assert False, "{} not supported".format(compression)


# https://docs.python.org/3/library/fnmatch.html
def scan_path(path, ignore=[], level=None, include_dirs=False, include_files=True, root=None):
    _is_dir_and_exist(path)
    is_recursive = level is None or level > 1
    root = path if root is None else root
    for entry in os.scandir(path):
        fp = _np(entry.path)
        rp = fp[len(root) + 1 :]
        if any(fnmatch.fnmatch(rp, pattern) for pattern in ignore):
            continue
        is_dir = entry.is_dir()
        n = fp.split("/")[-1]

        if include_dirs and include_files:
            yield fp, rp, n, is_dir
        elif include_dirs and is_dir:
            yield fp, rp, n
        elif include_files and not is_dir:
            yield fp, rp, n

        if is_recursive and is_dir:
            new_level = None if level is None else level - 1
            for t in scan_path(fp, ignore, new_level, include_dirs, include_files, root):
                yield t


def iter_txt(path, encoding="utf-8", first_n=None, sample_p=1.0, sample_seed=None, report_n=None, compression=None):
    with _open(path, encoding, compression) as f:
        for line in iterate(f, first_n, sample_p, sample_seed, report_n):
            yield line


def load_txt(path, encoding="utf-8", compression=None):
    return "".join(iter_txt(path, encoding=encoding, compression=compression))


def iter_jsonl(path, encoding="utf-8", first_n=None, sample_p=1.0, sample_seed=None, report_n=None, compression=None):
    with _open(path, encoding, compression) as f:
        for line in iterate(f, first_n, sample_p, sample_seed, report_n):
            yield json.loads(line)


def load_jsonl(path, encoding="utf-8", first_n=None, sample_p=1.0, sample_seed=None, report_n=None, compression=None):
    return list(iter_jsonl(path, encoding, first_n, sample_p, sample_seed, report_n, compression))


def load_json(path, encoding="utf-8", compression=None):
    with _open(path, encoding, compression) as f:
        return json.load(f)


def load_yaml(path, encoding="utf-8", compression=None):
    with _open(path, encoding, compression) as f:
        res = safe_load(f)
        return {} if res is None else res


def save_txt(path, data, encoding="utf-8"):
    with open(path, "w", encoding=encoding) as file:
        file.write(data)


def save_yaml(path, data, encoding="utf-8"):
    with open(path, "w", encoding=encoding) as file:
        dump(data, file, allow_unicode=True, sort_keys=False)


def save_jsonl(path, data, encoding="utf-8"):
    with open(path, "w", encoding=encoding) as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def save_json(path, data, indent=4, encoding="utf-8"):
    with open(path, "w", encoding=encoding) as f:
        return json.dump(data, f, indent=indent, ensure_ascii=False)


def type_of(data, types):
    assert isinstance(types, list), "types must be a list"
    for idx, _type in enumerate(types):
        if isinstance(data, _type):
            return idx + 1
    return 0


COLLECTION_TYPE = [list, set, tuple, dict]


def _build_table(rows, space=3, cell_space=1, filler=" ", max_column_width=None, min_column_widths=None):
    space = max(space, 1)

    t = type_of(rows, COLLECTION_TYPE)
    if not t:
        return [str(rows)]
    elif t == 4:
        temp = []
        for k, v in rows.items():
            r = _build_table(v, cell_space, cell_space, filler)
            temp.append([k, r])
        rows = temp

    # calculate max column width
    num_col = -1
    temp = []
    for r in rows:
        if not type_of(r, COLLECTION_TYPE):
            r = [r]
        num_col = max(num_col, len(r))
        temp.append(r)
    rows = temp

    data = []
    for r in rows:
        if len(r) != num_col:
            r += [""] * (num_col - len(r))
        r = [_build_table(x, cell_space, cell_space, filler) if type_of(x, COLLECTION_TYPE) else [str(x)] for x in r]
        max_height = max(len(r) for r in r)
        temp = [["" for _ in range(len(r))] for _ in range(max_height)]
        for j, items in enumerate(r):
            for i in range(len(items)):
                temp[i][j] = items[i]
        data.extend(temp)

    col_widths = [0 for _ in range(num_col)]
    for d in data:
        for i in range(num_col):
            if max_column_width is not None and len(d[i]) > max_column_width - 3:
                d[i] = shorten_str(d[i], max_column_width)
            col_widths[i] = max(col_widths[i], wcswidth(d[i]))
            if min_column_widths is not None and i < len(min_column_widths) and min_column_widths[i] is not None:
                col_widths[i] = max(col_widths[i], min_column_widths[i])

    res = []
    for d in data:
        r = []
        for idx in range(num_col - 1):
            i = d[idx]
            r.append(i)
            r.append(filler * (space + col_widths[idx] - wcswidth(i)))
        r.append(d[-1])
        res.append("".join(r))
    return res


def print_table(
    rows,
    headers=None,
    name="",
    sep="-",
    space=3,
    cell_space=1,
    filler=" ",
    max_column_width=None,
    min_column_widths=None,
    level=INFO,
    res=False,
    color=None,
):
    if headers is not None:
        rows = [headers] + rows
    _res = _build_table(rows, space, cell_space, filler, max_column_width, min_column_widths)
    first_sep_line = print_line(text=name, width=len(_res[0]), char=sep, res=True)
    sep_line = print_line(width=max(len(first_sep_line), len(_res[0])), char=sep, res=True)
    if headers is not None:
        _res = [first_sep_line, _res[0], sep_line] + _res[1:] + [sep_line]
    if not res:
        print_iter(_res, level=level, color=color)
    else:
        return _res


def _prints(data, indent, width, level, shift, extra_indent, sep, quote, kv_sep, compact, color):
    """
    extra_indent == None,  shift
    extra_indent == 0,     no shift
    extra_indent > 0,      no shift + shorter line
    """
    shift_str = shift * " "
    sep_len = len(sep)
    kv_sep_len = len(kv_sep)

    # int, float, single-line str
    def is_short_data(_d):
        if _d is None:
            return True
        r = type_of(_d, [int, float, str, bool])
        if r == 3:
            return not any(True for ch in _d if ch == "\n")
        return r

    def put_quote(string):
        return quote + string + quote if isinstance(string, str) else str(string)

    def log_raw(*args, **kwargs):
        kwargs["level"] = level
        kwargs["end"] = ""
        kwargs["color"] = color
        log(*args, **kwargs)

    def print_cache(_tokens, _shift, _extra_indent):
        line = sep.join(_tokens)
        if _extra_indent is None:
            line = _shift * " " + line
        log_raw(line)

    data_type = type_of(data, [list, tuple, set, dict, str])
    if is_short_data(data):
        if extra_indent is None:
            log_raw(shift_str + put_quote(data))
        else:
            log_raw(put_quote(data))
    # collection
    elif data_type in {1, 2, 3}:
        left, right = "[", "]"
        if data_type == 2:
            left, right = "(", ")"
        elif data_type == 3:
            left, right = "{", "}"
            data = list(data)
        if extra_indent is None:
            left = shift_str + left

        # handle empty string
        if not data:
            return log_raw(left + right)

        cache_size = 0 if extra_indent is None else extra_indent
        cache = []
        log_raw(left)
        # group data
        for idx, d in enumerate(data):
            if is_short_data(d):
                str_d = put_quote(d)
                if cache_size + len(str_d) + sep_len > width:
                    cache.append([])
                    cache_size = 0
                if not cache or not isinstance(cache[-1], list):
                    cache.append([])
                cache[-1].append(str_d)
                cache_size += len(str_d) + sep_len
            else:
                cache.append(idx)
                cache_size = 0
        # log
        for idx, d in enumerate(cache):
            if isinstance(d, list):
                print_cache(d, shift + 1, 0 if idx == 0 else None)
            else:
                _prints(data[d], indent, width, level, shift + 1, 0 if idx == 0 else None, sep, quote, kv_sep, compact, color)
            if idx != len(cache) - 1:
                log_raw("{}\n".format(sep))
        log_raw(right)
    # dictionary
    elif data_type == 4:
        left, right = "{", "}"
        if extra_indent is None:
            left = shift_str + left
        if not data:
            return log_raw(left + right)
        log_raw(left + "\n")

        kv = data.items()
        indent_str = indent * " "
        for idx, (k, v) in enumerate(kv):
            str_k = put_quote(k)
            if is_short_data(v):
                log_raw("{}{}{}{}".format(shift_str + indent_str, str_k, kv_sep, put_quote(v)))
            else:
                log_raw("{}{}{}".format(shift_str + indent_str, str_k, kv_sep))
                # for non-compact
                if isinstance(v, dict):
                    v_shift = shift + indent
                    v_indent = 0
                else:
                    if compact:
                        v_shift = shift + indent * 2 + 1
                        v_indent = max(0, len(str_k) + kv_sep_len - indent - 1)
                    else:
                        v_shift = shift + indent + len(str_k) + kv_sep_len
                        v_indent = 0
                _prints(v, indent, width, level, v_shift, v_indent, sep, quote, kv_sep, compact, color)
            if idx != len(kv) - 1:
                log_raw(sep + "\n")
            else:
                log_raw("\n")
        log_raw(shift_str + right)
    # multi-lines string
    elif data_type == 5:
        lines = data.split("\n")
        for idx, line in enumerate(lines):
            if idx == 0 and extra_indent is None:
                log_raw("{}".format(shift_str))
            elif idx == len(lines) - 1 and not line:
                continue
            elif idx != 0 or extra_indent is None:
                log_raw("\n{}".format(shift_str))
            log_raw("{}{}{}{}".format(quote, line, "\\n" if idx != len(lines) - 1 else "", quote))
    else:
        data = str(data)
        _prints(data, indent, width, level, shift, extra_indent, sep, quote, kv_sep, compact, color)


def prints(
    *data, indent=4, width=80, shift=0, extra_indent=None, compact=False, sep=",", quote='"', kv_sep=": ", level=INFO, res=False, color=None
):
    if res:
        with recorder() as r:
            for d in data:
                _prints(d, indent, width, level, shift, extra_indent, sep, quote, kv_sep, compact, color)
        return r.flush()
    else:
        for d in data:
            _prints(d, indent, width, level, shift, extra_indent, sep, quote, kv_sep, compact, color)
            log("", level=level)


def print_iter(data, shift=0, level=INFO, color=None):
    if not isinstance(data, Iterable):
        log(shift * " ", end="", color=color)
        log(data, level=level, color=color)
    else:
        if shift <= 0:
            for item in data:
                log(item, level=level, color=color)
        else:
            for item in data:
                log(shift * " ", end="", color=color)
                log(item, level=level, color=color)


def break_str(string, width=50, measure_f=wcswidth):
    res = [[]]
    curr = 0
    for ch in string:
        item_width = measure_f(ch)
        if curr + item_width <= width:
            res[-1].append(ch)
            curr += item_width
        elif curr == 0:
            res[-1].append(ch)
            res.append([])
        else:
            res.append([])
            res[-1].append(ch)
            curr = item_width
    res = [r for r in res if r]
    assert sum(len(r) for r in res) == len(string)
    return ["".join(r) for r in res]


def shorten_str(string, width=50):
    if len(string) <= width:
        return string
    return string[: (width - 3)] + "..."


def fill_str(string, left_marker="{", right_marker="}", **kwargs):
    for k, v in kwargs.items():
        string = string.replace(left_marker + k + right_marker, str(v))
    return string


VALID_REFERENCE_ARGUMENTS_PATTERN = r"\(([_a-zA-Z][_a-zA-Z0-9]*( *= *[_a-zA-Z0-9]+)?( *, *)?)+\)"


def debug(*data, mode=prints, char="-", level=DEBUG, color="red"):
    if LOGGER.level <= level:
        stack = inspect.stack()
        lineno = " [{}]".format(stack[1].lineno)
        filename = file_basename(stack[1][1]).split(".")[0]
        function_name = ".{}".format(stack[1][3]) if stack[1][3] != "<module>" else ""

        code_str = stack[1].code_context[0].strip()
        arguments = code_str[code_str.index("(") + 1 : -1]
        arguments = [a.strip() for a in arguments.split(",") if "=" not in a]
        assert len(data) == len(arguments), '{} ==> debug() can not take arguments with "," in it'.format(code_str)
        argument_str = "" if len(arguments) > 1 else ": {}".format(arguments[0])

        with block("{}{}{}{}".format(filename, function_name, lineno, argument_str), char=char, color=color):
            if mode is None:
                if len(data) > 1:
                    rows = []
                    for k, v in zip(arguments, data):
                        rows.append([k + ": ", str(v).split("\n")])
                    print_table(rows, space=1)
                else:
                    data = data[0]
                    log(data) if isinstance(data, str) else prints(data)
            elif mode == log or mode == print or mode == print_table:
                if len(data) > 1:
                    rows = []
                    for k, v in zip(arguments, data):
                        rows.append([k + ": ", str(v).split("\n")])
                    print_table(rows, space=1)
                else:
                    log(data[0])
            elif mode == print_iter:
                if len(data) > 1:
                    for k, v in zip(arguments, data):
                        log(k, end=": \n")
                        print_iter(v, shift=4)
                else:
                    print_iter(data[0])
            elif mode == prints:
                if len(data) > 1:
                    for k, v in zip(arguments, data):
                        log(k, end=": ")
                        prints(v, shift=len(k) + 2, extra_indent=0)
                else:
                    data = data[0]
                    log(data) if isinstance(data, str) else prints(data)
            else:
                assert False, "mode: {} not supported".format(mode)


def dmerge(*dicts):
    res = {}
    for d in dicts:
        for k, v in d.items():
            if k in res and isinstance(res[k], dict) and isinstance(v, dict):
                res[k] = dmerge(res[k], v)
            else:
                res[k] = v
    return res


def dsel(d, *args):
    return {k: d[k] for k in args if k in d}


def ddrop(d, *args):
    return {k: v for k, v in d.items() if k not in args}


def drename(d, k1, k2):
    return {(k2 if k == k1 else k): v for k, v in d.items()}


def try_f(*args, **kwargs):
    try:
        f = args[0]
        return {"res": f(*args[1:], **kwargs)}
    except Exception as e:
        return summarize_exception(e)


def stats_of(data, key_f=None, first_n=None, sample_p=1.0, sample_seed=None):
    _min, _max, _sum = float("inf"), -float("inf"), 0.0
    mean = 0.0
    sum_squared_deviations = 0.0
    iterator = iterate(data, first_n=first_n, sample_p=sample_p, sample_seed=sample_seed)
    if key_f is not None:
        iterator = map(key_f, iterator)

    counter = 0
    for x in iterator:
        counter += 1
        _min = min(_min, x)
        _max = max(_max, x)
        _sum += x

        delta = x - mean
        mean += delta / counter
        delta2 = x - mean
        sum_squared_deviations += delta * delta2

    var = sum_squared_deviations / counter if counter > 0 else 0.0
    std = var**0.5

    return {"count": counter, "sum": _sum, "mean": mean, "min": _min, "max": _max, "var": var, "std": std}


def _strip_and_add_spaces(s):
    return f" {s.strip()} " if s else ""


def print_line(text="", width=20, char="-", level=INFO, min_margin=5, res=False, color=None):
    if isinstance(text, int):
        if isinstance(width, str):
            width, text = text, width
        else:
            width, text = text, ""
    if text == "":
        chars = char * width
    else:
        text = _strip_and_add_spaces(text)
        margin = (width - len(text)) // 2
        margin = max(margin, min_margin)
        wing = char * margin
        chars = wing + text + wing
        if len(chars) < width:
            chars += char
    if res:
        return chars
    log(chars, level=level, color=color)


class block:
    def __init__(self, text="", width=None, max_width=80, char="=", end="\n\n", captured_level=INFO, level=INFO, color=None):
        self.text = _strip_and_add_spaces(text)
        self.width = width
        self.max_width = max_width
        self.char = char
        self.end = end
        self.level = level
        self.color = color
        if self.width is None:
            self.recorder = recorder(captured_level=captured_level)

    def __enter__(self):
        if self.width is not None:
            top_line = print_line(text=self.text, width=self.width, char=self.char, res=True)
            self.top_line_size = len(top_line)
            log(top_line, level=self.level, color=self.color)
        else:
            self.recorder.__enter__()

    def __exit__(self, *args):
        if self.width is None:
            self.recorder.__exit__(*args)
            content = self.recorder.flush()
            # enclosed lines should be slightly longer than the longest content
            content_width = 0 if not content else max(len(line) for line in content.split("\n"))
            content_width = min(self.max_width, content_width + 3)
            top_line = print_line(text=self.text, width=content_width, char=self.char, res=True)
            self.top_line_size = len(top_line)
            log(top_line, level=self.level, color=self.color)
            log(content, level=self.level, end="", color=self.color)
        log(print_line(width=self.top_line_size, char=self.char, level=self.level, res=True), end=self.end, color=self.color)


class block_timer:
    def __init__(self, text="", width=None, max_width=80, char="=", end="\n\n", captured_level=INFO, level=INFO, color=None):
        self._block = block(text, width, max_width, char, "\n", captured_level, level, color)
        self.end = end
        self.level = level

    def __enter__(self):
        self._block.__enter__()
        self.time_start = time.time()

    def __exit__(self, *args):
        time_end = time.time()
        self._block.__exit__(*args)
        log("took {:.3f} ms".format((time_end - self.time_start) * 1000), end=self.end, level=self.level)


def get_args(*args, **kwargs):
    p = argparse.ArgumentParser()
    seen = set()

    def check_arg(k):
        if k in seen:
            raise ValueError(f"Duplicated arg: {k}")
        elif k.startswith("-"):
            raise ValueError(f"Invalid arg: {k}")
        seen.add(k)

    def str2bool(x):
        return (
            x
            if isinstance(x, bool)
            else {"1": True, "true": True, "yes": True, "y": True, "0": False, "false": False, "no": False, "n": False}[x.lower()]
        )

    for k in args:
        if not isinstance(k, str):
            raise TypeError(f"Required arg must be str, got {type(k).__name__}")
        check_arg(k)
        p.add_argument(k, type=str)
    for k, v in kwargs.items():
        check_arg(k)
        if isinstance(v, bool):
            p.add_argument(f"--{k}", dest=k, nargs="?", type=str2bool, const=True, default=v)
            p.add_argument(f"--no-{k}", dest=k, action="store_false")
            p.add_argument(k, nargs="?", type=str2bool, default=argparse.SUPPRESS)
        elif isinstance(v, list):
            t = type(v[0]) if v else str
            p.add_argument(f"--{k}", dest=k, nargs="+", type=t, default=v)
            p.add_argument(k, nargs="*", type=t, default=argparse.SUPPRESS)
        else:
            t = type(v)
            p.add_argument(f"--{k}", dest=k, type=t, default=v)
            p.add_argument(k, nargs="?", type=t, default=argparse.SUPPRESS)
    return p.parse_args()


def run_with_args(entrypoint="main"):
    m = inspect.getmodule(inspect.stack()[1].frame) or sys.modules.get("__main__")
    funcs = {n: f for n, f in inspect.getmembers(m, inspect.isfunction) if f.__module__ == m.__name__ and not n.startswith("_")}
    argv = sys.argv
    if len(argv) < 2:
        fn_name, fn_argv = entrypoint, []
    else:
        if argv[1].startswith("-"):
            fn_name, fn_argv = entrypoint, argv[1:]
        else:
            fn_name, *fn_argv = argv[1:]

    if fn_name not in funcs:
        raise SystemExit(f"No such function: {fn_name} ==> options: {', '.join(funcs.keys())}")
    fn = funcs[fn_name]
    sig = inspect.signature(fn)
    kwargs = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect._empty}
    args = [k for k, v in sig.parameters.items() if v.default is inspect._empty]
    sys.argv = [argv[0]] + fn_argv
    res = fn(**vars(get_args(*args, **kwargs)))
    if inspect.isawaitable(res):
        asyncio.run(res)


def _np(path):
    return path.replace(os.sep, "/")


def npath(path):
    if path.startswith("~"):
        path = os.path.expanduser(path)
    return os.path.abspath(path).replace(os.sep, "/")


def jpath(*args, **kwargs):
    return _np(os.path.join(*args, **kwargs))


def lib_path():
    return _np(str(Path(__file__).absolute()))


def run_dir():
    return _np(os.getcwd())


def _is_file_and_exist(path):
    if os.path.exists(path):
        assert os.path.isfile(path), "{} ==> already exist but it's a directory".format(path)


def _is_dir_and_exist(path):
    if os.path.exists(path):
        assert os.path.isdir(path), "{} ==> already exist but it's a file".format(path)


def build_dirs(path_or_paths, overwrite=False):
    paths = path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
    for path in paths:
        path = Path(os.path.abspath(path))
        _is_dir_and_exist(path)
        if overwrite and os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def build_files(path_or_paths, overwrite=False):
    paths = path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
    for path in paths:
        _is_file_and_exist(path)
        if overwrite and os.path.exists(path):
            os.remove(path)
        build_dirs_for(path)
        open(path, "a").close()


def build_dirs_for(path_or_paths, overwrite=False):
    paths = path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
    for path in paths:
        path = Path(os.path.abspath(path))
        _is_file_and_exist(path)
        build_dirs(dir_of(path), overwrite)


def file_basename(path):
    path = Path(os.path.abspath(path))
    _is_file_and_exist(path)
    return os.path.basename(path)


def dir_basename(path):
    path = Path(os.path.abspath(path))
    _is_dir_and_exist(path)
    return os.path.basename(path)


def traverse(path, up=0, to=None, should_exist=False):
    if isinstance(up, str):
        should_exist = to if isinstance(to, bool) else should_exist
        to = up
        up = 0
    res = Path(os.path.abspath(path))
    o_res = res
    up = max(up, 0)
    for _ in range(up):
        n_res = res.parent
        assert n_res != res, "{} (went up {} times) ==> already reach root and cannot go up further".format(o_res, up)
        res = n_res
    res = str(res)
    if to is not None:
        res = jpath(res, to)
    assert not should_exist or os.path.exists(res), "{} ==> does not exist".format(res)
    return _np(res)


def dir_of(path):
    return traverse(path, 1)


def this_dir(up=0, to=None, should_exist=False):
    if isinstance(up, str):
        should_exist = to if isinstance(to, bool) else should_exist
        to = up
        up = 0
    caller_module = inspect.getmodule(inspect.stack()[1][0])
    return traverse(caller_module.__file__, up + 1, to, should_exist)


def unwrap_file(path):
    if os.path.isdir(path):
        sub_paths = os.listdir(path)
        assert len(sub_paths) == 1, "there are more than one files/dirs in {}".format(path)
        return unwrap_file(jpath(path, sub_paths[0]))
    return _np(path)


def unwrap_dir(path):
    if os.path.isdir(path):
        sub_paths = os.listdir(path)
        if len(sub_paths) == 1 and os.path.isdir(jpath(path, sub_paths[0])):
            return unwrap_dir(jpath(path, sub_paths[0]))
        return _np(path)
    assert False, "{} is not a directory".format(path)
