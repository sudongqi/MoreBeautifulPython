import re
import sys
import os
import random
import json
import time
import gzip
import bz2
import inspect
import itertools
import traceback
from datetime import datetime, timezone
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path

VERSION = '1.2.7'

__all__ = [
    # Alternative for multiprocessing
    'Workers', 'work',
    # Alternative for logging
    'log', 'logger', 'get_logger', 'set_global_logger', 'reset_global_logger',
    'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'SILENT',
    # Syntax sugar for pathlib
    'dir_of', 'path_join', 'make_dir', 'make_dir_for', 'this_dir', 'exec_dir', 'lib_path', 'only_file_of',
    # Tools for file loading & handling
    'load_jsonl', 'load_json', 'load_txt',
    'iterate', 'save_json', 'save_jsonl', 'open_file', 'file_paths_of', 'open_files',
    # Tools for summarizations
    'enclose', 'enclose_timer', 'error_msg',
    'prints', 'print_line', 'print_table', 'build_table', 'print_iter',
    # Tools for simple statistics
    'timer', 'curr_date_time', 'avg', 'min_max_avg', 'n_min_max_avg', 'CPU_COUNT'
]

NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL, SILENT = 0, 10, 20, 30, 40, 50, 60
CPU_COUNT = cpu_count()


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
            make_dir_for(f)
            res[i] = open(f, 'w', encoding='utf-8')
    return res


class Logger:
    def __init__(self, name='', file=sys.stdout, level=INFO, meta_info=False, sep=' '):
        self.level = level
        self.file = open_files_for_logger(file)
        self.prefix = name
        self.meta_info = True if name else meta_info
        self.sep = sep

    def __call__(self, msg, level=INFO, file=None, end=None, flush=False):
        if self.level <= level:
            _file = self.file if file is None else open_files_for_logger(file)
            for f in _file:
                if self.meta_info:
                    headers = [curr_date_time(), get_msg_level(level)]
                    if self.prefix:
                        headers.append(self.prefix)
                    print(self.sep.join(headers), file=f, end=': ', flush=flush)
                print(msg, file=f, end=end, flush=flush)


LOG = Logger()
CONTEXT_LOGGER_SET = False


class logger(object):
    def __init__(self, name='', file=sys.stdout, level=INFO, meta_info=False, can_overwrite=True):
        global LOG
        global CONTEXT_LOGGER_SET
        self.logger_was_changed = False
        if not CONTEXT_LOGGER_SET or not can_overwrite:
            self.org_logger = LOG
            LOG = Logger(name, file, level, meta_info)
            self.logger_was_changed = True
            CONTEXT_LOGGER_SET = True

    def __enter__(self):
        pass

    def __exit__(self, _type, value, _traceback):
        global LOG
        global CONTEXT_LOGGER_SET
        if self.logger_was_changed:
            LOG = self.org_logger
            CONTEXT_LOGGER_SET = False


def set_global_logger(name='', file=sys.stdout, level=INFO, meta_info=False, sep=' '):
    global LOG
    LOG = Logger(name, file, level, meta_info, sep)


def reset_global_logger():
    global LOG
    LOG = Logger()


def get_logger(name='', file=sys.stdout, level=INFO, meta_info=False, sep=' '):
    return Logger(name, file, level, meta_info, sep)


def curr_date_time():
    return str(datetime.now(timezone.utc))[:19]


def log(msg, level=INFO, file=None, end=None, flush=False):
    LOG(msg, level, file, end, flush)


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def make_dir_for(file_path):
    os.makedirs(dir_of(file_path), exist_ok=True)


def error_msg(e, detailed=True, seperator='\n'):
    if not detailed:
        return repr(e)
    else:
        res = traceback.format_exc()
        return res.replace('\n', seperator)


class Worker(Process):
    def __init__(self, f, inp, out, worker_id=None, cached_inp=None, built_inp=None, detailed_error=True,
                 progress=True):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.inp = inp
        self.out = out
        self.f = f
        self.cached_inp = cached_inp
        self.built_inp = None if built_inp is None else {k: v() for k, v in built_inp.items()}
        self.detailed_error = detailed_error
        if progress:
            log('started worker-{}'.format('?' if worker_id is None else worker_id))

    def run(self):
        while True:
            task_id, kwargs = self.inp.get()
            try:
                if isinstance(kwargs, dict):
                    if self.cached_inp is not None:
                        kwargs.update(self.cached_inp)
                    if self.built_inp is not None:
                        kwargs.update(self.built_inp)
                    res = self.f(**kwargs)
                else:
                    res = self.f(*kwargs)
                self.out.put({'worker_id': self.worker_id, 'task_id': task_id, 'res': res})
            except Exception as e:
                self.out.put({'worker_id': self.worker_id, 'task_id': task_id, 'res': None,
                              'error': error_msg(e, self.detailed_error)})


class Workers:
    def __init__(self, f, num_workers=CPU_COUNT, cached_inp=None, built_inp=None, progress=True, ignore_error=False):
        self.inp = Queue()
        self.out = Queue()
        self.workers = []
        self.task_id = 0
        self.progress = progress
        self.ignore_error = ignore_error
        self.f = f
        for i in range(num_workers):
            worker = Worker(f, self.inp, self.out, i, cached_inp, built_inp, not ignore_error, progress)
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
                yield self.get_res()
                running_task_num -= 1
        except StopIteration:
            for i in range(running_task_num):
                yield self.get_res()

    def map(self, tasks, ordered=False, res_only=True):
        if ordered:
            saved = {}
            id_task_waiting_for = 0
            for d in self._map(tasks):
                saved[d['task_id']] = d
                while id_task_waiting_for in saved:
                    if res_only:
                        yield saved[id_task_waiting_for]['res']
                    else:
                        yield saved[id_task_waiting_for]
                    saved.pop(id_task_waiting_for)
                    id_task_waiting_for += 1
        else:
            for d in self._map(tasks):
                if res_only:
                    yield d['res']
                else:
                    yield d

    def add_task(self, inp):
        self.inp.put((self.task_id, inp))
        self.task_id += 1

    def get_res(self):
        res = self.out.get()
        if 'error' in res:
            err_msg = 'worker-{} failed task-{} : {}'.format(res['worker_id'], res['task_id'], res['error'])
            if not self.ignore_error:
                self.terminate()
                assert False, err_msg
            if self.progress:
                log(err_msg)
        if self.progress:
            log('worker-{} completed task-{}'.format(res['worker_id'], res['task_id']))
        return res

    def terminate(self):
        for w in self.workers:
            w.terminate()
        if self.progress:
            log('terminated {} workers'.format(len(self.workers)))


def work(f, tasks, num_workers=CPU_COUNT, cached_inp=None, built_inp=None, progress=False, ordered=False,
         res_only=True, ignore_error=False):
    workers = Workers(f, num_workers, cached_inp, built_inp, progress, ignore_error)
    for d in workers.map(tasks, ordered, res_only):
        yield d
    workers.terminate()


class timer(object):
    def __init__(self, msg='', level=INFO):
        self.start = None
        self.msg = msg.strip()
        self.level = level

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, _type, value, _traceback):
        log('{}took {:.3f} ms'.format('' if self.msg == '' else self.msg + ' ==> ', (time.time() - self.start) * 1000),
            level=self.level)


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
                curr_time = time.time()
                speed = report_n / (curr_time - prev_time) if curr_time - prev_time != 0 else 'inf'
                log('{}/{} ==> {:.3f} items/s'.format(counter, total, speed))
                prev_time = curr_time


def open_file(path, encoding='utf-8', compression=None):
    if compression is None:
        return open(path, 'r', encoding=encoding)
    elif compression == 'gz':
        return gzip.open(path, 'rt', encoding=encoding)
    elif compression == 'bz2':
        return bz2.open(path, 'rb')
    else:
        assert False, '{} not supported'.format(compression)


def file_paths_of(path, pattern=".*\..*", progress=False):
    matcher = re.compile(pattern)
    for p, dirs, files in os.walk(path):
        for file_name in files:
            if matcher.fullmatch(file_name):
                file_path = path_join(p, file_name)
                try:
                    yield file_path
                    if progress:
                        log('found {} <== {}'.format(file_name, file_path))
                except PermissionError:
                    if progress:
                        log('no permission to open {} <== {}'.format(file_name, file_path))


def open_files(path, encoding='utf-8', compression=None, pattern=".*\..*", progress=False):
    for file_path in file_paths_of(path, pattern, progress):
        yield open_file(file_path, encoding, compression)


def save_json(data, path, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as f:
        return json.dump(data, f)


def save_jsonl(data, path, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


def load_json(path, encoding='utf-8', compression=None):
    with open_file(path, encoding, compression) as f:
        return json.load(f)


def load_jsonl(path, encoding="utf-8", first_n=None, sample_p=1.0, sample_seed=None, report_n=None, compression=None):
    with open_file(path, encoding, compression) as f:
        for line in iterate(f, first_n, sample_p, sample_seed, report_n):
            yield json.loads(line)


def load_txt(path, encoding="utf-8", first_n=None, sample_p=1.0, sample_seed=None, report_n=None, compression=None):
    with open_file(path, encoding, compression) as f:
        for line in iterate(f, first_n, sample_p, sample_seed, report_n):
            yield line.rstrip()


def build_table(rows, column_names=None, space=3):
    assert space >= 1, 'column_gap_size must be >= 1'

    rows = [[str(r) for r in row] for row in rows]

    num_col = None
    for row in rows:
        if num_col is None:
            num_col = len(row)
        else:
            assert num_col == len(row), 'rows have different size'

    if column_names is not None:
        rows = [column_names] + [[str(r) for r in row] for row in rows]

    sizes = [0] * num_col
    for row in rows:
        assert len(row) <= num_col
        for i, item in enumerate(row):
            sizes[i] = max(sizes[i], len(item))

    res = []
    for row in rows:
        stuff = []
        for i in range(num_col - 1):
            stuff.append(row[i])
            stuff.append(' ' * (space + sizes[i] - len(row[i])))
        stuff.append(row[-1])
        line = ''.join(stuff)
        res.append(line)
    return res


def print_table(rows, column_names=None, space=3, level=INFO):
    print_iter(build_table(rows, column_names, space), level=level)


def type_in(data, types):
    if isinstance(types, list):
        for idx, _type in enumerate(types):
            if isinstance(data, _type):
                return idx
    return None


def prints(data, indent=4, width=80, shift=0, sep=',', quote='"', level=INFO):
    _prints(data, indent, width, level, shift, None, sep, quote)
    log('', level=level)


def _prints(data, indent=4, width=80, level=INFO, shift=0, extra_indent=None, sep=',', quote='"'):
    sep_length = len(sep)

    def short_data(_d):
        r = type_in(_d, [int, float, str])
        if r == 2:
            return not any(True for ch in _d if ch == '\n')
        return r is not None

    def _s(string):
        return quote + string + quote if isinstance(string, str) else str(string)

    def _log(*args, **kwargs):
        kwargs['level'] = level
        kwargs['end'] = ''
        log(*args, **kwargs)

    shift_str = shift * ' '

    collections_type = type_in(data, [list, tuple, set])
    if short_data(data):
        _log(shift_str + _s(data))
    elif collections_type is not None:
        marker_l, marker_r = '[', ']'
        if collections_type == 1:
            marker_l, marker_r = '(', ')'
        elif collections_type == 2:
            marker_l, marker_r = '{', '}'

        tokens = []
        curr_line_count = [0 if extra_indent is None else extra_indent]
        first_line = [True]

        def flush_line_str(_tokens):
            line = ''.join(_tokens)
            if first_line[0]:
                line = line
                first_line[0] = False
            else:
                line = shift_str + ' ' + line
            _log(line)
            _tokens.clear()
            curr_line_count[0] = 0

        _log('{}{}'.format(('' if first_line[0] and extra_indent is not None else shift_str), marker_l))
        for idx, d in enumerate(data):
            if not short_data(d):
                if tokens:
                    flush_line_str(tokens)
                    _log('\n')
                if idx == 0:
                    _prints(d, indent, width, level, shift + 1, 0, sep, quote)
                else:
                    _prints(d, indent, width, level, shift + 1, None, sep, quote)
                if idx != len(data) - 1:
                    _log('{}\n'.format(sep))
            else:
                str_d = _s(d)
                if curr_line_count[0] > width:
                    flush_line_str(tokens)
                    _log('\n')
                curr_line_count[0] += len(str_d) + sep_length
                tokens.append(str_d)
                if idx != len(data) - 1:
                    tokens.append(sep)
        flush_line_str(tokens)
        _log(marker_r)
    elif isinstance(data, dict):
        first_line = [True]
        _log(('' if first_line[0] and extra_indent is not None else shift_str) + '{')
        kv = data.items()
        for idx, (k, v) in enumerate(kv):
            if short_data(v):
                _log('{}{}: {}{}'.format('\n' + shift_str + indent * ' ', _s(k), _s(v), sep))
            else:
                _log('{}{}: '.format('\n' + shift_str + indent * ' ', _s(k)))
                n_shift = shift + indent if isinstance(v, dict) else shift + indent * 2
                _prints(v, indent, width, level, n_shift, len(_s(k)) + 2 - indent, sep, quote)
                if idx != len(kv) - 1:
                    _log(sep)
        _log('\n' + shift_str + '}')
    elif isinstance(data, str):
        for s in data.split('\n'):
            _log('\n{}{}'.format(shift_str, quote + s + quote))


def print_iter(data, level=INFO):
    for item in data:
        log(item, level=level)


def n_min_max_avg(data, key_f=None, first_n=None, sample_p=1.0, sample_seed=None):
    res_min, res_max, res_sum = float('inf'), -float('inf'), 0
    iterator = iterate(data, first_n=first_n, sample_p=sample_p, sample_seed=sample_seed)
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


def strip_and_add_spaces(s):
    if s == '':
        return ''
    s = s.strip()
    if s[0] != ' ':
        s = ' ' + s
    if s[-1] != ' ':
        s = s + ' '
    return s


def print_line(text_or_length='', wing_size=10, char='-', level=INFO, no_print=False):
    if isinstance(text_or_length, str):
        wing = char * wing_size
        res = wing + strip_and_add_spaces(text_or_length) + wing
        if not no_print:
            log(res, level=level)
        return res
    elif isinstance(text_or_length, int):
        res = char * text_or_length
        if not no_print:
            log(res, level=level)
        return res
    else:
        assert False, 'text_or_length should be an instance of either str or int'


class enclose(object):
    def __init__(self, text_or_length='', wing_size=10, char='=', top_margin=0, bottom_margin=1, use_timer=False,
                 level=INFO):
        self.text_or_length = strip_and_add_spaces(text_or_length)
        self.wing_size = wing_size
        self.top_margin = top_margin
        self.bottom_margin = bottom_margin
        self.char = char
        self.start = None
        self.use_timer = use_timer
        self.level = level

    def __enter__(self):
        log('\n' * self.top_margin, end='', level=self.level)
        print_line(self.text_or_length, self.wing_size, self.char, self.level)
        self.start = time.time()

    def __exit__(self, _type, value, _traceback):
        log(self.char * (self.wing_size * 2 + len(self.text_or_length)), level=self.level)
        if self.use_timer:
            log('took {:.3f} ms'.format((time.time() - self.start) * 1000), level=self.level)
        log('\n' * self.bottom_margin, end='', level=self.level)


class enclose_timer(enclose):
    def __init__(self, text_or_length='', wing_size=10, char='=', top_margin=0, bottom_margin=1, level=INFO):
        super().__init__(text_or_length, wing_size, char, top_margin, bottom_margin, True, level)


def path_join(*args, **kwargs):
    return os.path.join(*args, **kwargs)


def lib_path():
    return str(Path(__file__).absolute())


def this_dir(move_up_or_sub_path=0, sub_path=None):
    caller_module = inspect.getmodule(inspect.stack()[1][0])
    if isinstance(move_up_or_sub_path, str):
        assert sub_path is None, 'if 1st argument is a string, 2nd argument should not be specified'
        return dir_of(caller_module.__file__, move_up=0, sub_path=move_up_or_sub_path)
    return dir_of(caller_module.__file__, move_up=move_up_or_sub_path, sub_path=sub_path)


def dir_of(file_path, move_up=0, sub_path=None):
    curr_path_obj = Path(file_path)
    for i in range(move_up + 1):
        curr_path_obj = curr_path_obj.parent
    res = str(curr_path_obj.absolute())
    if sub_path is not None:
        res = path_join(res, sub_path)
    return res


def only_file_of(dir_path):
    if os.path.isdir(dir_path):
        sub_paths = os.listdir(dir_path)
        assert len(sub_paths) == 1, 'there are more than one files/dirs in {}'.format(dir_path)
        return path_join(dir_path, sub_paths[0])
    return dir_path


def exec_dir():
    return os.getcwd()


if __name__ == '__main__':
    with enclose('More Beautiful Python', 30):
        _rows = [
            ['examples', 'https://github.com/sudongqi/MoreBeautifulPython/blob/main/examples.py'],
            ['execution_directory', exec_dir()],
            ['library_path', lib_path()],
            ['cpu_count', CPU_COUNT],
            ['version', VERSION]
        ]
        print_table(_rows)
