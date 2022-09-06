import sys
import time
import random
from src.mbp import *


# test function for multiprocessing
def test_f(x, fail_rate=0, running_time=0.2):
    time.sleep(running_time)
    assert random.random() > fail_rate, "simulated failure ({}%)".format(fail_rate * 100)
    return x * 2


# test function 2 for multiprocessing
def test_f2(idx, vec):
    return vec[idx]


def main():
    # all log() in this script will print to file at path "./log"
    set_global_logger(file=['./log', sys.stdout])
    # log() include all functionality of print()
    log('this is from the global logger')

    # get_logger() return a local logger just like logging.getLogger
    my_log = get_logger(__name__, meta_info=True)
    my_log('this is from the local logger', WARNING)
    '''
    2022-08-11 05:22:17 WARNING __main__: this is from the local logger
    '''

    # logger() context manager temporarily modify the global logger
    with logger(level=DEBUG, file=sys.stderr, name='__temp__', meta_info=True):
        # this message will be redirected to sys.stderr
        log('this is from the temporary logger', level=CRITICAL)
    '''
    2022-08-11 05:22:17 CRITICAL __temp__: this is from the temporary logger
    '''

    # suppress all logs by specifying level=SILENT
    with logger(level=SILENT):
        for i in range(1000):
            # all loggings are suppressed
            log(i)

    # context manager logger() will also overwrite all logger() within its scope
    with logger(level=INFO):
        with logger(level=SILENT):
            log('==> a hidden message')

    # except when can_overwrite == False
    with logger(level=INFO):
        with logger(level=SILENT, can_overwrite=False):
            log('==> this will never be printed')

    # use timer() context manager to get execution time
    with timer():
        d = {i: i for i in range(100)}
        for i in range(200000):
            d.get(i, None)
    '''
    took 6.980 ms
    '''

    # enclose() generate two text separators that enclose the execution
    with enclose(text_or_length='enclose()', wing_size=4, bottom_margin=1, char='=', use_timer=False):
        log('your first line')
        log('your second line')
    '''
    ==== enclose() ====
    your first line
    your second line
    ===================
    '''

    # enclose_timer() == enclose(timer=True)
    with enclose_timer():
        # iterate() can customize iteration procedures
        # for example, sample 10% and report every 3 yield from the first 100 samples
        for d in iterate(range(1000), first_n=100, sample_p=0.1, report_n=3):
            log(test_f(d, running_time=0.1))
    '''
    ====================
    2
    58
    66
    3/1000 ==> 9.369 items/s
    94
    104
    118
    6/1000 ==> 9.182 items/s
    126
    148
    184
    9/1000 ==> 9.144 items/s
    ====================
    took 974.987 ms
    '''

    # Workers() is more flexible than multiprocessing.Pool()
    n_task = 8
    with enclose_timer('Workers()'):
        workers = Workers(f=test_f, num_workers=4, progress=True, ignore_error=True)
        [workers.add_task({'x': i, 'fail_rate': 0.3}) for _ in range(n_task)]
        [workers.get_res() for _ in range(n_task)]
        workers.terminate()
    '''
    ========== Workers() ==========
    started worker-0
    started worker-1
    started worker-2
    started worker-3
    worker-1 completed task-1
    worker-0 completed task-0
    worker-3 completed task-3
    worker-2 completed task-2
    worker-3 completed task-6
    worker-0 completed task-5
    worker-1 failed task-4 : AssertionError('simulated failure (30.0%)')
    worker-1 completed task-4
    worker-2 failed task-7 : AssertionError('simulated failure (30.0%)')
    worker-2 completed task-7
    terminated 4 workers
    ===============================
    took 499.086 ms
    '''

    # similarly, we can use work() to process tasks from an iterator
    # tasks can be iterator of tuple (need to specify all inputs) or dict
    with enclose_timer('work()'):
        tasks = iter([(i, 0.5, 0.2) for i in range(n_task)])
        for r in work(f=test_f, tasks=tasks, ordered=True, ignore_error=True, res_only=False):
            log(r)
    '''
    ========== work() ==========
    {'worker_id': 1, 'task_id': 0, 'res': 0}
    {'worker_id': 0, 'task_id': 1, 'res': None, 'error': "AssertionError('simulated failure (50.0%)')"}
    {'worker_id': 4, 'task_id': 2, 'res': 4}
    {'worker_id': 2, 'task_id': 3, 'res': 6}
    {'worker_id': 3, 'task_id': 4, 'res': None, 'error': "AssertionError('simulated failure (50.0%)')"}
    {'worker_id': 5, 'task_id': 5, 'res': None, 'error': "AssertionError('simulated failure (50.0%)')"}
    {'worker_id': 6, 'task_id': 6, 'res': 12}
    {'worker_id': 8, 'task_id': 7, 'res': None, 'error': "AssertionError('simulated failure (50.0%)')"}
    ============================
    took 346.133 ms
    '''

    # use cached_inp = {'fixed_input': value, ...} to avoid pickling of heavy objects
    vec = [x for x in range(1000000)]
    with timer('work()'):
        a = list(work(test_f2, num_workers=1, ordered=True, tasks=iter((i, vec) for i in range(30))))
    with timer('work() with cached_inp'):
        b = list(work(test_f2, num_workers=1, ordered=True, tasks=iter({'idx': i} for i in range(30)),
                      cached_inp={'vec': vec}))
    assert a == b
    '''
    work() ==> took 1904.171 ms
    work() with cached_inp ==> took 100.659 ms
    '''

    def build_vec():
        return [x for x in range(1000000)]

    # for objects that can not be pickled, use built_inp
    with timer('work() with built_inp'):
        tasks = iter({'idx': i} for i in range(30))
        list(work(test_f2, num_workers=1, ordered=True, tasks=tasks, built_inp={'vec': build_vec}))
    '''
    work() with built_inp ==> took 128.602 ms
    '''

    with enclose('path'):
        # this_dir() return the directory of the current file
        log(this_dir())
        # path_join() == os.path.join()
        log(path_join(this_dir(), 'a', 'b', 'c.file'))
        # dir_of() find the directory of a file
        log(dir_of(__file__, move_up=2))
        # dir_of() can also extend a path
        log(dir_of(__file__, 0, 'hello.txt'))
        # exec_dir() return the directory where you run your python command
        log(exec_dir())
        # lib_path() return the path of the mbp library
        log(lib_path())
        # only_file_of() return the path of the only file in a folder
        log(only_file_of(this_dir(2, 'data')))
    '''
    ========== path ==========
    C:\\Users\sudon\MoreBeautifulPython
    C:\\Users\sudon\MoreBeautifulPython\a\b\c.file
    C:\\Users
    C:\\Users\sudon\MoreBeautifulPython\hello.txt
    C:\\Users\sudon\MoreBeautifulPython
    c:\\users\sudon\morebeautifulpython\src\mbp.py
    C:\\Users\data
    ==========================
    '''

    # open_files() return all files and their paths under a directory
    with enclose('open_files()'):
        for f in open_files(this_dir(), pattern='.*\.py', progress=True):
            pass
    '''
    ========== open_files() ==========
    found build_and_push.py <== C:\\Users\sudon\MoreBeautifulPython\build_and_push.py
    found examples.py <== C:\\Users\sudon\MoreBeautifulPython\examples.py
    found mbp.py <== C:\\Users\sudon\MoreBeautifulPython\src\mbp.py
    found __init__.py <== C:\\Users\sudon\MoreBeautifulPython\src\__init__.py
    ==================================
    '''

    jsonl_file_path = path_join(this_dir(), 'data.jsonl')
    json_file_path = path_join(this_dir(), 'data.json')

    # print_dict() is a superior pprint.pprint()
    prints(load_json(json_file_path), indent=4)
    '''
    {
        "quiz": {
            "maths": {
                "q1": {
                    "question": "5 + 7 = ?",
                    "options": ["10","11","12","13"],
                    "answer": "12",
                }
            }
        }
    }
    '''

    # other prints() examples, see log
    v_short = [1, 2, 3, 4, 5, "x", "y", "z"]
    v_long = [i for i in range(100)]
    v_hybrid = v_short + [v_long] + v_short
    v_multi = [v_short, v_short, v_short]
    v_nested = [[[0, 0], [1, 1], [2, 2]], [(0, 0), [1, 1], [2, 2]]]
    t_multi = [(1, 2, 3) for _ in range(3)]
    d1 = {'a': 'value', 'b': (1, 2, 3), 'c': v_short}
    d2 = {'a': 'value', 'a very ........... long key': v_long, 'c': set(v_short)}
    d3 = {'a': d1, 'b': d2}
    d4 = {'a': d3, 'hybrid': v_hybrid}
    num = 500
    comment = "this is a test string"
    comments = 'import sys\nimport os\nimport time\ndef test_f(a, b):\n\treturn a + b'
    d5 = {'single-line': comment, 'multi-line': comments}

    for x in [v_short, v_long, v_multi, v_hybrid, v_nested, t_multi, d1, d2, d3, d4, d5, num, comment]:
        prints(x, shift=5)

    # load_jsonl() return an iterator of dictionary
    data = list(load_jsonl(jsonl_file_path))

    # draw_line() will draw a line
    print_line()

    # print_iter() print items from an iterator one by one
    with enclose('print_list()'):
        print_iter(data)
    '''
    ========== print_list() ==========
    {'id': 1, 'name': 'Jackson', 'age': 43}
    {'id': 2, 'name': 'Zunaira', 'age': 24}
    {'id': 3, 'name': 'Lorelei', 'age': 72}
    ==================================
    '''

    # print_table() can adjust column width automatically
    # print_table(rows) == print_iter(build_table(rows))
    with enclose('print_table()'):
        rows = [d.values() for d in data]
        column_names = list(data[0].keys())
        print_table(rows, column_names=column_names, space=3)
    '''
    ========== print_table() ==========
    id   name      age
    1    Jackson   43
    2    Zunaira   24
    3    Lorelei   72
    ===================================
    '''

    # get 3 key statistics from an iterator at once
    with enclose_timer('simple statistics'):
        log(n_min_max_avg(load_jsonl(jsonl_file_path), key_f=lambda x: x['age']))
        log(min_max_avg(load_jsonl(jsonl_file_path), key_f=lambda x: x['age']))
        log(avg(load_jsonl(jsonl_file_path), key_f=lambda x: x['age']))
    '''
    ========== simple statistics ==========
    (3, 24, 72, 46.333333333333336)
    (24, 72, 46.333333333333336)
    46.333333333333336
    =======================================
    '''


if __name__ == '__main__':
    main()

