import os
import csv
import random
import json
import time
import gzip
import bz2
import itertools
import traceback
from multiprocessing import Process, Queue
from pathlib import Path
from tqdm import tqdm


def test_f(x, fail_rate=0, exec_time=0):
    assert random.random() > fail_rate, "simulated failure ({}%)".format(fail_rate * 100)
    time.sleep(exec_time)
    return x + 1


def error_msg(e, detailed=True, seperator='\n'):
    return repr(e) + seperator + repr(traceback.format_exc()) if detailed else repr(e)


class Worker(Process):
    def __init__(self, f, inp, out, worker_id=None, progress=True, detailed_error=False):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.inp = inp
        self.out = out
        self.f = f
        self.detailed_error = detailed_error
        if progress and worker_id is not None:
            print('started worker-{}'.format(worker_id))

    def run(self):
        while True:
            task_id, args = self.inp.get()
            try:
                res = self.f(*args)
                self.out.put({'worker_id': self.worker_id, 'task_id': task_id, 'res': res})
            except Exception as e:
                self.out.put({'worker_id': self.worker_id, 'task_id': task_id, 'res': None,
                              'error': error_msg(e, self.detailed_error)})


class Workers:
    def __init__(self, f, num_workers, progress=True, detailed_error=False):
        self.inp = Queue()
        self.out = Queue()
        self.workers = []
        self.progress = progress
        self.task_id = 0
        for i in range(num_workers):
            worker = Worker(f, self.inp, out=self.out, worker_id=i, progress=progress,
                            detailed_error=detailed_error)
            worker.start()
            self.workers.append(worker)

    def add_task(self, inp):
        self.inp.put((self.task_id, inp))
        self.task_id += 1

    def get_res(self):
        res = self.out.get()
        if self.progress:
            if 'error' in res:
                print('worker-{} failed task-{} : {}'.format(res['worker_id'], res['task_id'], res['error']))
            else:
                print('worker-{} completed task-{}'.format(res['worker_id'], res['task_id']))
        return res

    def terminate(self):
        for w in self.workers:
            w.terminate()


class timer(object):
    def __init__(self):
        self.start = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, _type, value, _traceback):
        print('took {} seconds'.format(time.time() - self.start))


def iterate(data, take_n=None, sample_ratio=1.0, sample_seed=None, progress=False):
    '''
    base iterator handler
    :param data: iterator
    :param take_n: stop the iterator after taking the first n items
    :param sample_ratio: probability of keep an item
    :param sample_seed: seed for sampling
    :param progress: should use tqdm or not
    :return: iteraotr
    '''
    if sample_seed is not None:
        random.seed(sample_seed)
    if take_n is not None:
        assert take_n >= 1, 'take_n should be >= 1'
    for d in tqdm(itertools.islice(data, 0, take_n), disable=not progress):
        if random.random() <= sample_ratio:
            yield d


def open_file(path, encoding='utf-8', compression=None):
    if compression is None:
        return open(path, 'r', encoding=encoding)
    elif compression == 'gz':
        return gzip.open(path, 'rt', encoding=encoding)
    elif compression == 'bz2':
        return bz2.open(path, 'rb')
    else:
        assert False, '{} not supported'.format(compression)


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


def load_jsonl(path, encoding="utf-8", take_n=None, sample_ratio=1.0, sample_seed=None, progress=False,
               compression=None):
    with open_file(path, encoding, compression) as f:
        for line in iterate(f, take_n, sample_ratio, sample_seed, progress):
            yield json.loads(line)


def load_txt(path, encoding="utf-8", take_n=None, sample_ratio=1.0, sample_seed=None, progress=False,
             compression=None):
    with open_file(path, encoding, compression) as f:
        for line in iterate(f, take_n, sample_ratio, sample_seed, progress):
            yield line.rstrip()


def load_csv(path, encoding="utf-8", delimiter=',', take_n=None, sample_ratio=1.0, sample_seed=None, progress=False,
             compression=None):
    csv.field_size_limit(10000000)
    with open_file(path, encoding, compression) as f:
        for d in iterate(csv.reader(f, delimiter=delimiter), take_n, sample_ratio, sample_seed, progress):
            yield d


def load_tsv(path, encoding="utf-8", take_n=None, sample_ratio=1.0, sample_seed=None, progress=False, compression=None):
    for d in load_csv(path, encoding, '/t', take_n, sample_ratio, sample_seed, progress, compression):
        yield d


def build_table(rows, column_names=None, columns_gap_size=3):
    assert columns_gap_size >= 1, 'column_gap_size must be >= 1'

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
            stuff.append(' ' * (columns_gap_size + sizes[i] - len(row[i])))
        stuff.append(row[-1])
        line = ''.join(stuff)
        res.append(line)
    return res


def print2(data, indent=4):
    print(json.dumps(data, indent=indent))


def print_list(data):
    for item in data:
        print(item)


def print_table(rows, column_names=None, columns_gap_size=3):
    print_list(build_table(rows, column_names, columns_gap_size))


def n_min_max_avg(data, key_f=None, take_n=None, sample_ratio=1.0, sample_seed=None):
    res_min, res_max, res_sum = float('inf'), -float('inf'), 0
    iterator = iterate(data, take_n=take_n, sample_ratio=sample_ratio, sample_seed=sample_seed, progress=False)
    if key_f is not None:
        iterator = map(key_f, iterator)
    counter = 0
    for num in iterator:
        res_min = min(res_min, num)
        res_max = max(res_max, num)
        res_sum += num
        counter += 1
    return counter, res_min, res_max, res_sum / counter


def min_max_avg(data, key_f=None, take_n=None, sample_ratio=1.0, sample_seed=None):
    return tuple(n_min_max_avg(data, key_f, take_n, sample_ratio, sample_seed)[1:])


def sep(text, size=10, char='-'):
    wing = char * size
    print(wing + text + wing)


class text_block(object):
    def __init__(self, text, size=10, char='-', y_gap_size=1):
        self.text = text
        self.size = size
        self.char = char
        self.y_gap_size = y_gap_size

    def __enter__(self):
        print('\n' * self.y_gap_size, end="")
        sep(self.text, self.size, self.char)

    def __exit__(self, _type, value, _traceback):
        print(self.char * (self.size * 2 + len(self.text)))
        print('\n' * self.y_gap_size, end="")


def path_join(path, *paths):
    return os.path.join(path, *paths)


def lib_path():
    return str(Path(__file__).absolute())


def dir_of(file, level=1):
    curr_path_obj = Path(file)
    for i in range(level):
        curr_path_obj = curr_path_obj.parent
    return str(curr_path_obj.absolute())


def get_only_file(path):
    if os.path.isdir(path):
        sub_paths = os.listdir(path)
        assert len(sub_paths) == 1, 'there are more than one files/dirs in {}'.format(path)
        return path_join(path, sub_paths[0])
    return path


def exec_dir():
    return os.getcwd()


if __name__ == '__main__':
    with text_block('examples', 29, '=', 2):
        print('https://github.com/sudongqi/AbsolutelyEssentialToolKit/examples.py')
