import re
import sys
import os
import shutil
import random
import json
import yaml
import time
import gzip
import bz2
import inspect
import itertools
import traceback
import argparse
import queue
from io import StringIO
from collections.abc import Iterator, Iterable
from datetime import datetime, timezone
from multiprocessing import cpu_count
from threading import Thread
from pathlib import Path
from wcwidth import wcswidth


__all__ = [
    # replacement for logging
    'log', 'context_logger', 'local_logger', 'set_global_logger', 'global_logger_level', 'reset_global_logger', 'recorder',
    # logging levels
    'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'SILENT',
    # replacement for multiprocessing
    'Workers', 'work',
    # syntax sugar for common utilities
    'try_f', 'stop', 'type_of', 'range_of', 'items_of', 'npath', 'jpath', 'run_dir', 'lib_path',
    # handling data files
    'load_lines', 'load_txt', 'load_jsonl', 'load_json', 'load_yaml', 'save_txt', 'save_json', 'save_jsonl', 'save_yaml', 'iterate', 'open_file',
    # handling paths
    'unwrap_file', 'unwrap_dir', 'file_basename', 'dir_basename',
    # tools for file system
    'traverse', 'this_dir', 'dir_of', 'build_dirs', 'build_files', 'build_dirs_for', 'iterate_files', 'open_files',
    # handling string
    'break_str', 'shorten_str', 'fill_str',
    # tools for debug
    'enclose', 'enclose_timer', 'error_msg', 'debug',
    # tools for summarizations
    'prints', 'print_iter', 'print_table', 'print_line',
    # tools for simple statistics
    'timer', 'curr_time', 'avg', 'min_max_avg', 'n_min_max_avg', 'CPU_COUNT', 'MIN', 'MAX',
    # tools for environment
    'env', 'load_env', 'get_args', 'parse_args', 'args_to_cmd'
]

NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL, SILENT = 0, 10, 20, 30, 40, 50, 60
CPU_COUNT = cpu_count()
MIN = float('-inf')
MAX = float('inf')


def get_msg_level(level):
    if level < 10:
        return 'NOTSET'
    elif level < 20:
        return 'DEBUG'
    elif level < 30:
        return 'INFO'
    elif level < 40:
        return 'WARNING'
    elif level < 50:
        return 'ERROR'
    elif level < 60:
        return 'CRITICAL'
    else:
        return 'SILENT'


def open_files_for_logger(file):
    res = file if isinstance(file, list) else [file]
    for i in range(len(res)):
        f = res[i]
        if isinstance(f, str):
            build_dirs_for(f)
            res[i] = open(f, 'w', encoding='utf-8')
    return res


class Logger:
    def __init__(self, name='', file=sys.stdout, level=INFO, verbose=False):
        self.level = level
        self.file = open_files_for_logger(file)
        self.name = name
        self.verbose = True if name else verbose

    def __call__(self, *data, level=INFO, file=None, end=None, flush=True):
        if self.level <= level:
            header = f'{curr_time()} {get_msg_level(level)} {self.name}: ' \
                if self.name != '' else f'{curr_time()} {get_msg_level(level)}: '
            header_empty = len(header) * ' '
            for f in (self.file if file is None else open_files_for_logger(file)):
                for d in data:
                    lines = str(d).split('\n')
                    for idx, line in enumerate(lines):
                        if self.verbose:
                            if idx == 0:
                                print(header, file=f, end='', flush=flush)
                            else:
                                print(header_empty, file=f,
                                      end='', flush=flush)
                        if idx == len(lines) - 1:
                            print(line, file=f, end=end, flush=flush)
                        else:
                            print(line, file=f, end=None, flush=flush)


LOGGER = Logger()
CONTEXT_LOGGER_SET = False


class context_logger(object):
    def __init__(self, name='', file=sys.stdout, level=INFO, verbose=False, can_overwrite=True):
        global LOGGER
        global CONTEXT_LOGGER_SET
        self.logger_was_changed = False
        if not CONTEXT_LOGGER_SET or not can_overwrite:
            self.org_logger = LOGGER
            LOGGER = Logger(name, file, level, verbose)
            self.logger_was_changed = True
            CONTEXT_LOGGER_SET = True

    def __enter__(self):
        pass

    def __exit__(self, *args):
        global LOGGER
        global CONTEXT_LOGGER_SET
        if self.logger_was_changed:
            LOGGER = self.org_logger
            CONTEXT_LOGGER_SET = False


def set_global_logger(name='', file=sys.stdout, level=INFO, verbose=False):
    global LOGGER
    LOGGER = Logger(name, file, level, verbose)


def global_logger_level():
    global LOGGER
    return LOGGER.level


def reset_global_logger():
    global LOGGER
    LOGGER = Logger()


def local_logger(name='', file=sys.stdout, level=INFO, verbose=False):
    return Logger(name, file, level, verbose)


def curr_time(breakdown=False):
    res = str(datetime.now(timezone.utc))[:19]
    if breakdown:
        #      year           month          day             hour             minute           second
        return int(res[0:4]), int(res[5:7]), int(res[8:10]), int(res[11:13]), int(res[14:16]), int(res[17:19])
    return res


def log(*messages, level=INFO, file=None, end=None, flush=True):
    LOGGER(*messages, level=level, file=file, end=end, flush=flush)


class recorder(object):
    def __init__(self, tape, captured_level=INFO):
        assert tape == [], '1st argument must be an empty list'
        self.buffer = StringIO()
        self.logger = context_logger(
            file=self.buffer, level=captured_level, can_overwrite=False)
        self.tape = tape

    def __enter__(self):
        self.logger.__enter__()

    def __exit__(self, _type, value, _traceback):
        buffer_value = self.buffer.getvalue()
        if buffer_value:
            buffer_value = buffer_value.split('\n')
            self.tape.extend(buffer_value)
        self.logger.__exit__(_type, value, _traceback)


def error_msg(e, verbose=False, sep='\n'):
    if not verbose:
        return repr(e)
    else:
        res = traceback.format_exc()
        return _np(res.replace('\n', sep))


class Worker(Thread):
    def __init__(self, f, tasks, results, worker_id=None, cache_inp=None, build_inp=None, verbose=True):
        super(Worker, self).__init__(daemon=True)
        self.worker_id = worker_id
        self.tasks = tasks
        self.results = results
        self.f = f
        self.cache_inp = cache_inp
        self.built_inp = build_inp
        if verbose:
            log('started worker-{}'.format('?' if worker_id is None else worker_id))

    def run(self):
        self.built_inp = None if self.built_inp is None else {k: v[0](*v[1:]) for k, v in self.built_inp.items()}
        while True:
            task_id, kwargs = self.tasks.get()
            if task_id is None:
                break
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
                self.results.put({'worker_id': self.worker_id,
                                  'task_id': task_id,
                                  'inp': kwargs,
                                  'res': res})
            except Exception as e:
                self.results.put(
                    {'worker_id': self.worker_id,
                     'task_id': task_id,
                     'inp': kwargs,
                     'error': error_msg(e, False),
                     'traceback': error_msg(e, True)})


class Workers:
    def __init__(self, f, num_workers=CPU_COUNT, cache_inp=None, build_inp=None, ignore_error=False, verbose=True):
        self.tasks = queue.Queue()
        self.results = queue.Queue()
        self.workers = []
        self.task_count = 0
        self.verbose = verbose
        self.ignore_error = ignore_error
        self.f = f
        for i in range(num_workers):
            worker = Worker(f, self.tasks, self.results, i, cache_inp, build_inp, verbose)
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
                saved[d['task_id']] = d
                while id_task_waiting_for in saved:
                    yield saved[id_task_waiting_for]
                    saved.pop(id_task_waiting_for)
                    id_task_waiting_for += 1
        else:
            for d in self._map(tasks):
                yield d

    def add_task(self, task):
        assert isinstance(task, tuple) or isinstance(task, dict), "task must be tuple of dict"
        self.tasks.put((self.task_count, task))
        self.task_count += 1

    def get_result(self):
        res = self.results.get()
        if 'error' in res:
            error_message = 'worker-{} failed task-{} : {}'.format(res['worker_id'], res['task_id'], res['error'])
            if not self.ignore_error:
                self.terminate()
                assert False, error_message
            if self.verbose:
                log(error_message)
        elif self.verbose:
            log('worker-{} completed task-{}'.format(res['worker_id'], res['task_id']))
        return res

    def terminate(self):
        for _ in self.workers:
            self.tasks.put((None, None))
        for worker in self.workers:
            worker.join()
        if self.verbose:
            log('terminated {} workers'.format(len(self.workers)))


def work(f, tasks, num_workers=CPU_COUNT, cache_inp=None, build_inp=None,
         ordered=True, ignore_error=False, res_only=True, verbose=False):
    workers = Workers(f, num_workers, cache_inp,
                      build_inp, ignore_error, verbose)
    for d in workers.map(tasks, ordered):
        yield d.get('res', None) if res_only else d
    workers.terminate()


class timer(object):
    def __init__(self, text='', level=INFO):
        self.start = None
        self.msg = text.strip()
        self.level = level

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, _type, value, _traceback):
        duration = (time.time() - self.start) * 1000
        log('{}took {:.3f} ms'.format('' if self.msg ==
            '' else self.msg + ' ==> ', duration), level=self.level)


def iterate(data, first_n=None, sample_p=1.0, sample_seed=None, report_n=None):
    if sample_seed is not None:
        random.seed(sample_seed)
    if first_n is not None:
        assert first_n >= 1, 'first_n should be >= 1'
    counter = 0
    total = len(data) if hasattr(data, '__len__') else '?'
    prev_time = time.time()
    for d in itertools.islice(data, 0, first_n):
        if random.random() <= sample_p:
            counter += 1
            yield d
            if report_n is not None and counter % report_n == 0:
                current_time = time.time()
                speed = report_n / \
                    (current_time - prev_time) if current_time - \
                    prev_time != 0 else 'inf'
                log('{}/{} ==> {:.3f} items/s'.format(counter, total, speed))
                prev_time = current_time


def open_file(path, encoding='utf-8', compression=None):
    if compression is None:
        return open(path, 'r', encoding=encoding)
    elif compression == 'gz':
        return gzip.open(path, 'rt', encoding=encoding)
    elif compression == 'bz2':
        return bz2.open(path, 'rb')
    else:
        assert False, '{} not supported'.format(compression)


def iterate_files(path, pattern=r".*"):
    # return  iterator of (absolute_path, relative_path, file_name)
    _is_dir_and_exist(path)
    matcher = re.compile(pattern)
    for p, dirs, files in os.walk(path):
        for file_name in files:
            if matcher.fullmatch(file_name):
                full_path = jpath(p, file_name)
                yield _np(os.path.abspath(full_path)), _np(full_path[len(path):]), file_name


def open_files(path, encoding='utf-8', compression=None, pattern=r".*", verbose=True):
    for full_path, _, file_name in iterate_files(path, pattern):
        try:
            yield open_file(full_path, encoding, compression)
            if verbose:
                log('found {} <== {}'.format(file_name, full_path))
        except PermissionError:
            if verbose:
                log('no permission to open {} <== {}'.format(file_name, full_path))


def load_lines(path, encoding="utf-8", first_n=None, sample_p=1.0, sample_seed=None, report_n=None, compression=None):
    with open_file(path, encoding, compression) as f:
        for line in iterate(f, first_n, sample_p, sample_seed, report_n):
            yield line


def load_txt(path, encoding="utf-8", compression=None):
    return "".join(load_lines(path, encoding=encoding, compression=compression))


def load_jsonl(path, encoding="utf-8", first_n=None, sample_p=1.0, sample_seed=None, report_n=None, compression=None):
    with open_file(path, encoding, compression) as f:
        for line in iterate(f, first_n, sample_p, sample_seed, report_n):
            yield json.loads(line)


def load_json(path, encoding='utf-8', compression=None):
    with open_file(path, encoding, compression) as f:
        return json.load(f)


def load_yaml(path, encoding='utf-8', compression=None):
    with open_file(path, encoding, compression) as f:
        return yaml.safe_load(f)


def save_txt(path, data, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as file:
        file.write(data)


def save_yaml(path, data, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as file:
        yaml.dump(data, file, allow_unicode=True, sort_keys=False)


def save_jsonl(path, data, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


def save_json(path, data, indent=4, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as f:
        return json.dump(data, f, indent=indent)


def type_of(data, types):
    assert isinstance(types, list), 'types must be a list'
    for idx, _type in enumerate(types):
        if isinstance(data, _type):
            return idx + 1
    return 0


def _range_iterate(data, start, end=sys.maxsize, step=1):
    assert not end < 0, 'end cannot be negative'
    start = max(start, 0)
    step = max(step, 1)
    for idx, item in enumerate(data):
        if start <= idx < end and (idx - start) % step == 0:
            yield idx, item


def range_of(data, start=0, end=None, step=1, reverse=False):
    # replace of ==> for i in range(data)

    assert isinstance(data, Iterable), 'data should be an Iterable'
    if isinstance(data, Iterator):
        assert not reverse, 'cannot set reverse=True when data is an Iterator'
        for idx, _ in _range_iterate(data, start, end, step):
            yield idx
    else:
        step = max(step, 1)
        start = max(start, 0)
        data_len = len(data)
        if end is None:
            end = data_len
        elif end > 0:
            end = min(end, data_len)
        else:
            end = max(0, data_len + end)
        if reverse:
            for i in range(end - 1, start - 1, -step):
                yield i
        else:
            for i in range(start, end, step):
                yield i


def items_of(data, start=0, end=None, step=1, reverse=False):
    assert isinstance(data, Iterable), 'input should be an Iterable'
    if isinstance(data, Iterator):
        assert not reverse, 'cannot set reverse=True when data is an Iterator'
        for _, item in _range_iterate(data, start, end, step):
            yield item

    for idx in range_of(data, start, end, step, reverse):
        yield data[idx]


COLLECTION_TYPES = [list, set, tuple, dict]


def _build_table(rows, space=3, cell_space=1, filler=' ', max_column_width=None, min_column_widths=None):
    space = max(space, 1)

    rows_type = type_of(rows, COLLECTION_TYPES)
    if not rows_type:
        return [str(rows)]
    elif rows_type == 4:
        _rows = []
        for k, v in rows.items():
            r = _build_table(v, cell_space, cell_space, filler)
            _rows.append([k, r])
        rows = _rows

    # calculate max column width
    num_col = -1
    _rows = []
    for row in rows:
        row_type = type_of(row, COLLECTION_TYPES)
        if not row_type:
            row = [row]
        num_col = max(num_col, len(row))
        _rows.append(row)
    rows = _rows

    data = []
    for row in rows:
        if len(row) != num_col:
            row += [''] * (num_col - len(row))
        row = [_build_table(item, cell_space, cell_space, filler)
               if type_of(item, COLLECTION_TYPES) else [str(item)]
               for item in row]
        max_height = max(len(r) for r in row)
        new_rows = [['' for _ in range(len(row))] for _ in range(max_height)]
        for j, items in enumerate(row):
            for i in range(len(items)):
                new_rows[i][j] = items[i]
        data.extend(new_rows)

    column_width = [0 for _ in range(num_col)]
    for d in data:
        for i in range(num_col):
            if max_column_width is not None and len(d[i]) > max_column_width - 3:
                d[i] = shorten_str(d[i], max_column_width)
            column_width[i] = max(column_width[i], wcswidth(d[i]))
            if min_column_widths is not None and i < len(min_column_widths) and min_column_widths[i] is not None:
                column_width[i] = max(column_width[i], min_column_widths[i])

    res = []
    for d in data:
        row = []
        for idx in range(num_col - 1):
            item = d[idx]
            row.append(item)
            row.append(filler * (space + column_width[idx] - wcswidth(item)))
        row.append(d[-1])
        res.append(''.join(row))
    return res


def print_table(rows, headers=None, name='', sep='-', space=3, cell_space=1, filler=' ',
                max_column_width=None, min_column_widths=None, level=INFO, res=False):
    if headers is not None:
        rows = [headers] + rows
    _res = _build_table(rows, space, cell_space, filler,
                        max_column_width, min_column_widths)
    first_sep_line = print_line(text=name, width=len(_res[0]), char=sep, res=True)
    sep_line = print_line(width=max(len(first_sep_line), len(_res[0])), char=sep, res=True)
    if headers is not None:
        _res = [first_sep_line, _res[0], sep_line] + _res[1:] + [sep_line]
    if not res:
        print_iter(_res, level=level)
    else:
        return _res


def _prints(data, indent, width, level, shift, extra_indent, sep, quote, kv_sep, compact):
    """
    extra_indent == None,  shift
    extra_indent == 0,     no shift
    extra_indent > 0,      no shift + shorter line
    """
    shift_str = shift * ' '
    sep_len = len(sep)
    kv_sep_len = len(kv_sep)

    # int, float, single-line str
    def is_short_data(_d):
        if _d is None:
            return True
        r = type_of(_d, [int, float, str, bool])
        if r == 3:
            return not any(True for ch in _d if ch == '\n')
        return r

    def put_quote(string):
        return quote + string + quote if isinstance(string, str) else str(string)

    def log_raw(*args, **kwargs):
        kwargs['level'] = level
        kwargs['end'] = ''
        log(*args, **kwargs)

    def print_cache(_tokens, _shift, _extra_indent):
        line = sep.join(_tokens)
        if _extra_indent is None:
            line = _shift * ' ' + line
        log_raw(line)

    data_type = type_of(data, [list, tuple, set, dict, str])
    if is_short_data(data):
        if extra_indent is None:
            log_raw(shift_str + put_quote(data))
        else:
            log_raw(put_quote(data))
    # collection
    elif data_type in {1, 2, 3}:
        marker_l, marker_r = '[', ']'
        if data_type == 2:
            marker_l, marker_r = '(', ')'
        elif data_type == 3:
            marker_l, marker_r = '{', '}'
            data = list(data)
        if extra_indent is None:
            marker_l = shift_str + marker_l

        # handle empty string
        if not data:
            return log_raw(marker_l + marker_r)

        cache_size = 0 if extra_indent is None else extra_indent
        cache = []
        log_raw(marker_l)
        # group data
        for idx, d in enumerate(data):
            d_type = type(d)
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
                _prints(data[d], indent, width, level, shift + 1,
                        0 if idx == 0 else None, sep, quote, kv_sep, compact)
            if idx != len(cache) - 1:
                log_raw('{}\n'.format(sep))
        log_raw(marker_r)
    # dictionary
    elif data_type == 4:
        marker_l = '{'
        marker_r = '}'
        if extra_indent is None:
            marker_l = shift_str + marker_l

        if not data:
            return log_raw(marker_l + marker_r)
        log_raw(marker_l + '\n')

        kv = data.items()
        indent_str = indent * ' '
        for idx, (k, v) in enumerate(kv):
            str_k = put_quote(k)
            if is_short_data(v):
                log_raw('{}{}{}{}'.format(
                    shift_str + indent_str, str_k, kv_sep, put_quote(v)))
            else:
                log_raw('{}{}{}'.format(shift_str + indent_str, str_k, kv_sep))
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
                _prints(v, indent, width, level, v_shift,
                        v_indent, sep, quote, kv_sep, compact)
            if idx != len(kv) - 1:
                log_raw(sep + '\n')
            else:
                log_raw('\n')
        log_raw(shift_str + marker_r)
    # multi-lines string
    elif data_type == 5:
        _data = data.split('\n')
        for idx, s in enumerate(_data):
            if idx == 0 and extra_indent is None:
                log_raw('{}'.format(shift_str))
            elif idx == len(_data) - 1 and not s:
                continue
            elif idx != 0 or extra_indent is None:
                log_raw('\n{}'.format(shift_str))
            log_raw('{}{}{}{}'.format(quote, s, '\\n' if idx !=
                    len(_data) - 1 else '', quote))
    else:
        data = str(data)
        _prints(data, indent, width, level, shift,
                extra_indent, sep, quote, kv_sep, compact)


def prints(*data, indent=4, width=80, shift=0, extra_indent=None, compact=False, sep=',', quote='"', kv_sep=': ',
           level=INFO, res=False):
    if res:
        res = []
        with recorder(res):
            for d in data:
                _prints(d, indent, width, level, shift,
                        extra_indent, sep, quote, kv_sep, compact)
        return '\n'.join(res)
    else:
        for d in data:
            _prints(d, indent, width, level, shift,
                    extra_indent, sep, quote, kv_sep, compact)
            log('', level=level)


def print_iter(data, shift=0, level=INFO):
    if not isinstance(data, Iterable):
        log(shift * ' ', end='')
        log(data, level=level)
    else:
        if shift <= 0:
            for item in data:
                log(item, level=level)
        else:
            for item in data:
                log(shift * ' ', end='')
                log(item, level=level)


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
    return [''.join(r) for r in res]


def shorten_str(string, width=50):
    if len(string) <= width:
        return string
    return string[:(width - 3)] + '...'


def fill_str(string, left_marker="{", right_marker="}", **kwargs):
    for k, v in kwargs.items():
        string = string.replace(left_marker + k + right_marker, str(v))
    return string


def stop(message=''):
    raise SystemExit(message)


VALID_REFERENCE_ARGUMENTS_PATTERN = r'\(([_a-zA-Z][_a-zA-Z0-9]*( *= *[_a-zA-Z0-9]+)?( *, *)?)+\)'


def debug(*data, mode=prints, char='-', level=DEBUG):
    if LOGGER.level <= level:

        stack = inspect.stack()
        lineno = ' [{}]'.format(stack[1].lineno)
        filename = file_basename(stack[1][1]).split('.')[0]
        function_name = '.{}'.format(
            stack[1][3]) if stack[1][3] != '<module>' else ''

        code_str = stack[1].code_context[0].strip()
        arguments = code_str[code_str.index("(") + 1:-1]
        arguments = [a.strip() for a in arguments.split(',') if '=' not in a]
        assert len(data) == len(
            arguments), '{} ==> debug() can not take arguments with "," in it'.format(code_str)
        argument_str = '' if len(
            arguments) > 1 else ': {}'.format(arguments[0])

        with enclose('{}{}{}{}'.format(filename, function_name, lineno, argument_str), char=char):
            if mode is None:
                if len(data) > 1:
                    rows = []
                    for k, v in zip(arguments, data):
                        rows.append([k + ': ', str(v).split('\n')])
                    print_table(rows, space=1)
                else:
                    data = data[0]
                    log(data) if isinstance(data, str) else prints(data)
            elif mode == log or mode == print or mode == print_table:
                if len(data) > 1:
                    rows = []
                    for k, v in zip(arguments, data):
                        rows.append([k + ': ', str(v).split('\n')])
                    print_table(rows, space=1)
                else:
                    log(data[0])
            elif mode == print_iter:
                if len(data) > 1:
                    for k, v in zip(arguments, data):
                        log(k, end=': \n')
                        print_iter(v, shift=4)
                else:
                    print_iter(data[0])
            elif mode == prints:
                if len(data) > 1:
                    for k, v in zip(arguments, data):
                        log(k, end=': ')
                        prints(v, shift=len(k) + 2, extra_indent=0)
                else:
                    data = data[0]
                    log(data) if isinstance(data, str) else prints(data)
            else:
                assert False, 'mode: {} not supported'.format(mode)


def try_f(*args, **kwargs):
    res = {}
    try:
        f = args[0]
        res['res'] = f(*args[1:], **kwargs)
    except Exception as e:
        res['error'] = error_msg(e, verbose=False)
        res['error_type'] = res['error'].split('(')[0]
        res['error_msg'] = res['error'][len(res['error_type']) + 2: -2]
        res['traceback'] = error_msg(e, verbose=True)
    return res


def n_min_max_avg(data, key_f=None, first_n=None, sample_p=1.0, sample_seed=None):
    res_min, res_max, res_sum = float('inf'), -float('inf'), 0
    iterator = iterate(data, first_n=first_n,
                       sample_p=sample_p, sample_seed=sample_seed)
    if key_f is not None:
        iterator = map(key_f, iterator)
    counter = 0
    for num in iterator:
        res_min = min(res_min, num)
        res_max = max(res_max, num)
        res_sum += num
        counter += 1
    return counter, res_min, res_max, res_sum / counter


def min_max_avg(data, key_f=None, first_n=None, sample_p=1.0, sample_seed=None):
    return tuple(n_min_max_avg(data, key_f, first_n, sample_p, sample_seed)[1:])


def avg(data, key_f=None, first_n=None, sample_p=1.0, sample_seed=None):
    return n_min_max_avg(data, key_f, first_n, sample_p, sample_seed)[3]


def _strip_and_add_spaces(s):
    if s == '':
        return ''
    s = s.strip()
    if s[0] != ' ':
        s = ' ' + s
    if s[-1] != ' ':
        s = s + ' '
    return s


def print_line(text=None, width=20, char='-', level=INFO, min_wing_size=5, res=False):
    if text is None:
        _res = char * width
    else:
        text = _strip_and_add_spaces(text)
        wing_size = (width - len(text)) // 2
        wing_size = max(wing_size, min_wing_size)
        wing = char * wing_size
        _res = wing + text + wing
        if len(_res) < width:
            _res += char
    if not res:
        log(_res, level=level)
    else:
        return _res


class enclose(object):
    def __init__(self, text='', width=None, max_width=80, char='=', top_margin=0, bottom_margin=1, use_timer=False,
                 level=INFO, captured_level=INFO):
        self.text = _strip_and_add_spaces(text)
        self.width = width
        self.max_width = max_width
        self.top_margin = top_margin
        self.bottom_margin = bottom_margin
        self.char = char
        self.start = None
        self.use_timer = use_timer
        self.level = level
        self.aligned = width is None
        if self.aligned:
            self.tape = []
            self.recorder = recorder(self.tape, captured_level=captured_level)

    def __enter__(self):
        if not self.aligned:
            log('\n' * self.top_margin, end='', level=self.level)
            top_line = print_line(text=self.text, width=self.width, char=self.char, res=True)
            self.top_line_size = len(top_line)
            log(top_line, level=self.level)
        else:
            self.recorder.__enter__()
        self.start = time.time()

    def __exit__(self, _type, value, _traceback):
        if self.aligned:
            self.recorder.__exit__(_type, value, _traceback)
            max_line_length = len(self.text)
            if self.tape:
                # enclosed lines should be slightly longer than the longest content
                max_line_length = max(len(msg) + 3 for msg in self.tape)
            max_line_length = min(self.max_width, max_line_length)
            log('\n' * self.top_margin, end='', level=self.level)
            top_line = print_line(text=self.text, width=max_line_length, char=self.char, res=True)
            self.top_line_size = len(top_line)
            log(top_line, level=self.level)
            log('\n'.join(self.tape), level=self.level, end='')
        print_line(width=self.top_line_size, char=self.char, level=self.level)

        if self.use_timer:
            log('took {:.3f} ms'.format(
                (time.time() - self.start) * 1000), level=self.level)
        log('\n' * self.bottom_margin, end='', level=self.level)


class enclose_timer(enclose):
    def __init__(self, text='', width=None, max_width=80, char='=', top_margin=0, bottom_margin=1, level=INFO):
        super().__init__(text, width, max_width, char,
                         top_margin, bottom_margin, True, level)


def env(key, default_value=None):
    if default_value is None:
        return os.environ[key]
    return os.environ.get(key, default_value)


def load_env(dict_or_path):
    d = load_yaml(dict_or_path) if isinstance(dict_or_path, dict) else dict_or_path
    for k, v in d.items():
        os.environ[k] = str(v)


def get_args(*args, **kwargs):
    parser = argparse.ArgumentParser()
    for k in args:
        parser.add_argument(k, type=str)
    for k, v in kwargs.items():
        if isinstance(v, list):
            parser.add_argument(f"--{k}", nargs="+", default=v, type=type(v[0]))
        elif isinstance(v, bool):
            parser.add_argument(f"--{k}", default=v, type=type(v), action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser.parse_args()


def parse_args():
    args = []
    kwargs = {}
    vec = sys.argv[2:][::-1] if sys.argv[0] == '-m' else sys.argv[1:][::-1]
    while vec:
        if vec[-1].startswith("--"):
            k = vec.pop()[2:]
            assert len(vec) > 0, f"--{k} is expecting a value"
            v = vec.pop()
            kwargs[k] = v
        else:
            args.append(vec.pop())
    return args, kwargs


def args_to_cmd(args=[], kwargs={}):
    res = args[:]
    for k, v in kwargs.items():
        res.append(f"--{k}")
        res.append(v)
    return " ".join(res)


def _np(path):
    return path.replace(os.sep, '/')


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
        assert os.path.isfile(
            path), '{} ==> already exist but it\'s a directory'.format(path)


def _is_dir_and_exist(path):
    if os.path.exists(path):
        assert os.path.isdir(
            path), '{} ==> already exist but it\'s a file'.format(path)


def build_dirs(path_or_paths, overwrite=False):
    paths = path_or_paths if isinstance(
        path_or_paths, list) else [path_or_paths]
    for path in paths:
        path = Path(os.path.abspath(path))
        _is_dir_and_exist(path)
        if overwrite and os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def build_files(path_or_paths, overwrite=False):
    paths = path_or_paths if isinstance(
        path_or_paths, list) else [path_or_paths]
    for path in paths:
        _is_file_and_exist(path)
        if overwrite and os.path.exists(path):
            os.remove(path)
        build_dirs_for(path)
        open(path, 'a').close()


def build_dirs_for(path_or_paths, overwrite=False):
    paths = path_or_paths if isinstance(
        path_or_paths, list) else [path_or_paths]
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


def traverse(path, go_up=0, go_to=None, should_exist=False):
    if isinstance(go_up, str):
        should_exist = go_to if isinstance(go_to, bool) else should_exist
        go_to = go_up
        go_up = 0
    res = Path(os.path.abspath(path))
    o_res = res
    go_up = max(go_up, 0)
    for i in range(go_up):
        n_res = res.parent
        assert n_res != res, '{} (go up {} times) ==> already reach root and cannot go up further'.format(
            o_res, go_up)
        res = n_res
    res = str(res)
    if go_to is not None:
        res = jpath(res, go_to)
    assert not should_exist or os.path.exists(
        res), '{} ==> does not exist'.format(res)
    return _np(res)


def dir_of(path):
    return traverse(path, 1)


def this_dir(go_up=0, go_to=None, should_exist=False):
    if isinstance(go_up, str):
        should_exist = go_to if isinstance(go_to, bool) else should_exist
        go_to = go_up
        go_up = 0
    caller_module = inspect.getmodule(inspect.stack()[1][0])
    return traverse(caller_module.__file__, go_up + 1, go_to, should_exist)


def unwrap_file(path):
    if os.path.isdir(path):
        sub_paths = os.listdir(path)
        assert len(
            sub_paths) == 1, 'there are more than one files/dirs in {}'.format(path)
        return unwrap_file(jpath(path, sub_paths[0]))
    return _np(path)


def unwrap_dir(path):
    if os.path.isdir(path):
        sub_paths = os.listdir(path)
        if len(sub_paths) == 1 and os.path.isdir(jpath(path, sub_paths[0])):
            return unwrap_dir(jpath(path, sub_paths[0]))
        return _np(path)
    assert False, '{} is not a directory'.format(path)
