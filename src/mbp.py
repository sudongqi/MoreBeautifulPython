import re
import sys
import os
import csv
import random
import json
import time
import gzip
import glob
import bz2
import inspect
import itertools
import traceback
from datetime import datetime, timezone
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path

__all__ = [
    # Alternative for logging
    'log', 'logger', 'get_logger', 'set_global_logger', 'reset_global_logger',
    'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'SILENT',
    # Alternative for multiprocessing
    'Workers', 'work', 'test_f',
    # Syntax sugar for pathlib
    'dir_of', 'path_join', 'make_dir', 'this_dir', 'exec_dir', 'lib_path', 'only_file_of',
    # Tools for file loading & handling
    'load_jsonl', 'load_json', 'load_csv', 'load_tsv', 'load_txt',
    'iterate', 'save_json', 'save_jsonl', 'open_file', 'open_files',
    # Tools for summarizations
    'print2', 'log2', 'enclose', 'enclose_timer', 'print_table', 'build_table', 'print_iter', 'error_msg', 'sep', 'na',
    # Tools for simple statistics
    'timer', 'curr_time', 'avg', 'min_max_avg', 'n_min_max_avg', 'CPU_COUNT',
    # common libraries
    'sys', 'os', 'random', 'json', 'itertools',
]

NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL, SILENT = 0, 10, 20, 30, 40, 50, 100
CPU_COUNT = cpu_count()


class Logger:
    def __init__(self, file=sys.stdout, level=INFO, prefix='', meta_info=False, sep='  '):
        self.level = level
        self.file = None
        self.prefix = prefix
        self.meta_info = True if prefix else meta_info
        self.sep = sep
        self.direct_to(file)

    def direct_to(self, path):
        self.file = path
        if isinstance(path, str):
            make_dir(path)
            self.file = open(path, 'w', encoding='utf-8')

    def __call__(self, msg, level=INFO, file=None, end=None):
        if self.level <= level:
            _file = self.file if file is None else file
            if self.meta_info:
                print(curr_time(), file=_file, end=self.sep)
            if self.prefix:
                print(self.prefix, file=_file, end=self.sep)
            print(msg, file=_file, end=end)


LOG = Logger()


class logger(object):
    def __init__(self, file=sys.stdout, level=INFO, prefix='', meta_info=False):
        global LOG
        self.org_logger = LOG
        LOG = Logger(file, level, prefix, meta_info)

    def __enter__(self):
        pass

    def __exit__(self, _type, value, _traceback):
        global LOG
        LOG = self.org_logger


def set_global_logger(file=sys.stdout, level=INFO, prefix='', meta_info=False, sep='  '):
    global LOG
    LOG = Logger(file, level, prefix, meta_info, sep)


def reset_global_logger():
    global LOG
    LOG = Logger()


def get_logger(file=sys.stdout, level=INFO, prefix='', meta_info=False, sep='  '):
    return Logger(file, level, prefix, meta_info, sep)


def curr_time():
    return str(datetime.now(timezone.utc))[:19]


def log(msg, level=INFO, file=None, end=None):
    if LOG.level <= level:
        LOG(msg, level, file, end)


def make_dir(path):
    os.makedirs(dir_of(path), exist_ok=True)
    return path


def test_f(x, fail_rate=0, running_time=0.2):
    time.sleep(running_time)
    assert random.random() > fail_rate, "simulated failure ({}%)".format(fail_rate * 100)
    return x * 2


def error_msg(e, detailed=True, seperator='\n\n'):
    return repr(e) + seperator + traceback.format_exc() if detailed else repr(e)


class Worker(Process):
    def __init__(self, f, inp, out, worker_id=None, detailed_error=False, progress=True):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.inp = inp
        self.out = out
        self.f = f
        self.detailed_error = detailed_error
        if progress:
            log('started worker-{}'.format(na(worker_id)))

    def run(self):
        while True:
            task_id, kwargs = self.inp.get()
            try:
                res = self.f(**kwargs)
                self.out.put({'worker_id': self.worker_id, 'task_id': task_id, 'res': res})
            except Exception as e:
                self.out.put({'worker_id': self.worker_id, 'task_id': task_id, 'res': None,
                              'error': error_msg(e, self.detailed_error)})


class Workers:
    def __init__(self, f, num_workers=CPU_COUNT, detailed_error=False, progress=True):
        self.inp = Queue()
        self.out = Queue()
        self.workers = []
        self.task_id = 0
        self.progress = progress
        for i in range(num_workers):
            worker = Worker(f, self.inp, self.out, i, detailed_error, progress)
            worker.start()
            self.workers.append(worker)

    def map(self, data):
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
            self.terminate()

    def add_task(self, inp):
        self.inp.put((self.task_id, inp))
        self.task_id += 1

    def get_res(self):
        res = self.out.get()
        if self.progress:
            if 'error' in res:
                log('worker-{} failed task-{} : {}'.format(res['worker_id'], res['task_id'], res['error']))
            else:
                log('worker-{} completed task-{}'.format(res['worker_id'], res['task_id']))
        return res

    def terminate(self):
        for w in self.workers:
            w.terminate()
        if self.progress:
            log('terminated {} workers'.format(len(self.workers)))


def work(f, tasks, num_workers=CPU_COUNT, progress=False, ordered=False):
    if ordered:
        saved = {}
        id_task_waiting_for = 0
        for d in Workers(f, num_workers, progress=progress).map(tasks):
            saved[d['task_id']] = d
            while id_task_waiting_for in saved:
                yield saved[id_task_waiting_for]
                saved.pop(id_task_waiting_for)
                id_task_waiting_for += 1
    else:
        for d in Workers(f, num_workers, progress=progress).map(tasks):
            yield d


class timer(object):
    def __init__(self, level=INFO):
        self.start = None
        self.level = level

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, _type, value, _traceback):
        log('took {} seconds'.format(time.time() - self.start), level=self.level)


def iterate(data, first_n=None, sample_ratio=1.0, sample_seed=None, report_n=None):
    if sample_seed is not None:
        random.seed(sample_seed)
    if first_n is not None:
        assert first_n >= 1, 'first_n should be >= 1'
    counter = 0
    total = len(data) if hasattr(data, '__len__') else '?'
    for d in itertools.islice(data, 0, first_n):
        if random.random() <= sample_ratio:
            counter += 1
            yield d
            if report_n is not None and counter % report_n == 0:
                log('{}/{}'.format(counter, total))


def open_file(path, encoding='utf-8', compression=None):
    if compression is None:
        return open(path, 'r', encoding=encoding)
    elif compression == 'gz':
        return gzip.open(path, 'rt', encoding=encoding)
    elif compression == 'bz2':
        return bz2.open(path, 'rb')
    else:
        assert False, '{} not supported'.format(compression)


def open_files(path, encoding='utf-8', compression=None, pattern=".*\..*"):
    matcher = re.compile(pattern)
    for p, dirs, files in os.walk(path):
        for file_name in files:
            if matcher.fullmatch(file_name):
                file_path = path_join(p, file_name)
                try:
                    yield open_file(file_path, encoding, compression)
                    log('found {} <== {}'.format(file_name, file_path))
                except PermissionError:
                    log('no permission to open {} <== {}'.format(file_name, file_path))


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


def load_jsonl(path, encoding="utf-8", first_n=None, sample_ratio=1.0, sample_seed=None, progress_interval=None,
               compression=None):
    with open_file(path, encoding, compression) as f:
        for line in iterate(f, first_n, sample_ratio, sample_seed, progress_interval):
            yield json.loads(line)


def load_txt(path, encoding="utf-8", first_n=None, sample_ratio=1.0, sample_seed=None, progress_interval=None,
             compression=None):
    with open_file(path, encoding, compression) as f:
        for line in iterate(f, first_n, sample_ratio, sample_seed, progress_interval):
            yield line.rstrip()


def load_csv(path, encoding="utf-8", delimiter=',', first_n=None, sample_ratio=1.0, sample_seed=None,
             progress_interval=False, compression=None):
    csv.field_size_limit(10000000)
    with open_file(path, encoding, compression) as f:
        for d in iterate(csv.reader(f, delimiter=delimiter), first_n, sample_ratio, sample_seed, progress_interval):
            yield d


def load_tsv(path, encoding="utf-8", first_n=None, sample_ratio=1.0, sample_seed=None, progress_interval=None,
             compression=None):
    for d in load_csv(path, encoding, '/t', first_n, sample_ratio, sample_seed, progress_interval, compression):
        yield d


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


def log2(data, indent=4, level=INFO):
    log(json.dumps(data, indent=indent), level=level)


def print2(data, indent=4):
    print(json.dumps(data, indent=indent))


def print_iter(data, level=INFO):
    for item in data:
        log(item, level=level)


def n_min_max_avg(data, key_f=None, first_n=None, sample_ratio=1.0, sample_seed=None):
    res_min, res_max, res_sum = float('inf'), -float('inf'), 0
    iterator = iterate(data, first_n=first_n, sample_ratio=sample_ratio, sample_seed=sample_seed)
    if key_f is not None:
        iterator = map(key_f, iterator)
    counter = 0
    for num in iterator:
        res_min = min(res_min, num)
        res_max = max(res_max, num)
        res_sum += num
        counter += 1
    return counter, res_min, res_max, res_sum / counter


def min_max_avg(data, key_f=None, first_n=None, sample_ratio=1.0, sample_seed=None):
    return tuple(n_min_max_avg(data, key_f, first_n, sample_ratio, sample_seed)[1:])


def avg(data, key_f=None, first_n=None, sample_ratio=1.0, sample_seed=None):
    return n_min_max_avg(data, key_f, first_n, sample_ratio, sample_seed)[3]


def na(item, na_str='?'):
    return na_str if item is None else item


def sep(text='', size=10, char='=', level=INFO):
    wing = char * size
    log(wing + text + wing, level=level)


class enclose(object):
    def __init__(self, text='', size=10, margin=1, char='=', timer=False, level=INFO):
        self.text = text
        self.size = size
        self.size_y = margin
        self.char = char
        self.start = None
        self.timer = timer
        self.level = level

    def __enter__(self):
        sep(self.text, self.size, self.char, self.level)
        self.start = time.time()

    def __exit__(self, _type, value, _traceback):
        log(self.char * (self.size * 2 + len(self.text)), level=self.level)
        if self.timer:
            log('took {} seconds'.format(time.time() - self.start), level=self.level)
        log('\n' * self.size_y, end='', level=self.level)


class enclose_timer(enclose):
    def __init__(self, text='', size=10, margin=1, char='=', level=INFO):
        super().__init__(text, size, margin, char, True, level)


def path_join(path, *paths):
    return os.path.join(path, *paths)


def lib_path():
    return str(Path(__file__).absolute())


def this_dir(go_up=0):
    caller_module = inspect.getmodule(inspect.stack()[1][0])
    return dir_of(caller_module.__file__, go_up=go_up)


def dir_of(file, go_up=0):
    curr_path_obj = Path(file)
    for i in range(go_up + 1):
        curr_path_obj = curr_path_obj.parent
    return str(curr_path_obj.absolute())


def only_file_of(path):
    if os.path.isdir(path):
        sub_paths = os.listdir(path)
        assert len(sub_paths) == 1, 'there are more than one files/dirs in {}'.format(path)
        return path_join(path, sub_paths[0])
    return path


def exec_dir():
    return os.getcwd()


if __name__ == '__main__':
    with enclose('More Beautiful Python', 30):
        _rows = [
            ['example', 'https://github.com/sudongqi/MoreBeautifulPython/examples.py'],
            ['execution directory', exec_dir()],
            ['library path', lib_path()],
            ['cpu count', CPU_COUNT],
        ]
        print_table(_rows)
