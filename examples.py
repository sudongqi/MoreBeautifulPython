import sys
import shutil
import time
import random

# from mbp import * (for pip install version)
from src.mbp import *


# test functions for multiprocessing
def test_sleep_and_fail(x, fail_rate=0, running_time=0.2):
    time.sleep(running_time)
    assert random.random() > fail_rate, "simulated failure ({}%)".format(fail_rate * 100)
    return x


def test_read_from_list(idx, vec):
    return vec[idx]


def main():
    # log() include all functionality of print()
    log('this is from the global logger', end='\n')

    # from this point on, all log() will print to file at path "./log" as well as stdout
    set_global_logger(file=['./log', sys.stdout])

    # get_logger() return a local logger just like logging.getLogger
    my_log = get_logger(__name__, meta_info=True)
    my_log('this is from the local logger', WARNING)
    '''
    2022-08-11 05:22:17 WARNING __main__: this is from the local logger
    '''

    # logger() (as context manager) temporarily modify the global logger
    with logger(level=DEBUG, file=sys.stderr, name='__temp__', meta_info=True):
        # this message will be redirected to sys.stderr
        log('this is from the temporary logger', level=CRITICAL)
    '''
    2022-08-11 05:22:17 CRITICAL __temp__: this is from the temporary logger
    '''

    # suppress all logs by setting level=SILENT
    with logger(level=SILENT):
        for i in range(1000):
            # all loggings are suppressed
            log(i)

    # logger() will also overwrite all logger() within its scope
    with logger(level=INFO):
        with logger(level=SILENT):
            log('==> a hidden message')

    # except when can_overwrite == False
    with logger(level=INFO):
        with logger(level=SILENT, can_overwrite=False):
            log('==> this will never be printed')

    # print_line() will draw a line with am optional message
    print_line()
    print_line(text_or_width=30)
    print_line(text_or_width='optional message', width=30, char='=')
    '''
    --------------------
    ------------------------------
    ====== optional message ======
    '''

    # enclose() generate two text separators that enclose the execution
    with enclose('enclose()', width=30, bottom_margin=1, char='=', use_timer=False):
        [log('this is line {}'.format(i)) for i in range(3)]
    '''
    ========= enclose() ==========
    this is line 0
    this is line 1
    this is line 2
    ==============================
    '''

    # when width=None, the width will be calculated automatically based on the captured content
    with enclose():
        [log('this is line {}'.format(i)) for i in range(3)]
    '''
    ================
    this is line 0
    this is line 1
    this is line 2
    ================
    '''

    # recorder() save all logs into a list
    with enclose('recorder()'):
        tape = []
        with recorder(tape):
            log('9 8 7 6 5 4 3 2 1')
            log('ok', end='')
        tape[0] = tape[0][::-1]
        print_iter(tape)
    '''
    ===== recorder() =====
    1 2 3 4 5 6 7 8 9
    ok
    ======================
    '''

    # use timer() context manager to get execution time
    with timer():
        d = {i: i for i in range(100)}
        for i in range(200000):
            d.get(i, None)
    '''
    took 6.951 ms
    '''

    # enclose_timer() == enclose(timer=True)
    with enclose_timer():
        # iterate() can customize iteration procedures
        # for example, sample 10% and report every 3 yield from the first 100 samples
        for d in iterate(range(1000), first_n=100, sample_p=0.05, report_n=2):
            log(test_sleep_and_fail(d, running_time=0.1))
    '''
   ==========================
    18
    72
    2/1000 ==> 9.187 items/s
    78
    128
    4/1000 ==> 9.237 items/s
    ==========================
    took 435.241 ms
    '''

    # Workers() is more flexible than multiprocessing.Pool()
    n_task = 6
    with enclose_timer('Workers()'):
        workers = Workers(f=test_sleep_and_fail, num_workers=3, progress=True, ignore_error=True)
        [workers.add_task({'x': i, 'fail_rate': 0.3}) for _ in range(n_task)]
        [workers.get_res() for _ in range(n_task)]
        workers.terminate()
    '''
    ============================= Workers() ==============================
    started worker-0
    started worker-1
    started worker-2
    worker-2 failed task-2 : AssertionError('simulated failure (30.0%)')
    worker-1 failed task-1 : AssertionError('simulated failure (30.0%)')
    worker-0 completed task-0
    worker-2 completed task-4
    worker-0 completed task-5
    worker-1 failed task-3 : AssertionError('simulated failure (30.0%)')
    terminated 3 workers
    ======================================================================
    took 495.347 ms
    '''

    # similarly, we can use work() to process tasks from an iterator
    # tasks can be iterator of tuple (need to specify all inputs) or dict
    with enclose_timer('work()'):
        tasks = iter([(i, 0.5, 0.2) for i in range(n_task)])
        for r in work(f=test_sleep_and_fail, tasks=tasks, ordered=True, ignore_error=True, res_only=False):
            log(r)
    '''
    ============================================== work() ===============================================
    {'worker_id': 0, 'task_id': 0, 'res': 0}
    {'worker_id': 1, 'task_id': 1, 'res': 1}
    {'worker_id': 3, 'task_id': 2, 'res': 2}
    {'worker_id': 4, 'task_id': 3, 'res': None, 'error': "AssertionError('simulated failure (50.0%)')"}
    {'worker_id': 2, 'task_id': 4, 'res': 4}
    {'worker_id': 5, 'task_id': 5, 'res': 5}
    =====================================================================================================
    took 342.720 ms
    '''

    with enclose("work() with cache_inp"):
        # use cached_inp = {'fixed_input': value, ...} to avoid pickling of heavy objects
        vec = [x for x in range(1000000)]
        with timer('work()'):
            a = list(work(test_read_from_list, num_workers=1, ordered=True, tasks=iter((i, vec) for i in range(30))))
        with timer('work() with cached_inp'):
            b = list(work(test_read_from_list, num_workers=1, ordered=True, tasks=iter({'idx': i} for i in range(30)),
                          cached_inp={'vec': vec}))
        assert a == b

        def build_vec():
            return [x for x in range(1000000)]

        # for objects that can not be pickled, use built_inp
        with timer('work() with built_inp'):
            tasks = iter({'idx': i} for i in range(30))
            list(work(test_read_from_list, num_workers=1, ordered=True, tasks=tasks, built_inp={'vec': build_vec}))
    '''
    ========== work() with cache_inp ==========
    work() ==> took 1932.197 ms
    work() with cached_inp ==> took 96.938 ms
    work() with built_inp ==> took 129.255 ms
    ===========================================
    '''

    with enclose('pathing'):
        # path_join() == os.path.join()
        log(join_path(this_dir(), 'a', 'b', 'c.file'))
        # exec_dir() == os.getcwd()
        log(exec_dir())
        # lib_path() return the path of the mbp library
        log(lib_path())
        # this_dir() return the directory of the current file
        log(this_dir())
        log(this_dir(go_up=1, go_to='AnotherProject/hello.txt'))
        # only_file_of() return the path of the only file of the input directory
        log(get_only_file_if_dir(this_dir(2, 'data')))
        # dir_name_of() check and return the directory name of a path
        log(dir_name_of(dir_of(__file__)))
        # file_name_of() check and return the file name of a path
        log(file_name_of(__file__))
    '''
    =================== pathing ===================
    C:\\Users\sudon\MoreBeautifulPython\a\b\c.file
    C:\\Users\sudon\MoreBeautifulPython
    C:\\Users\sudon\MoreBeautifulPython\src\mbp.py
    C:\\Users\sudon\MoreBeautifulPython
    C:\\Users\sudon\AnotherProject/hello.txt
    C:\\Users\data
    MoreBeautifulPython
    examples.py
    ===============================================
    '''

    # open_files() return all files under a directory
    with enclose('open_files()'):
        for f in open_files(this_dir(), pattern=r'.*\.py'):
            f.readlines()
    '''
    ================================== open_files() ==================================
    found build_and_push.py <== C:\\Users\sudon\MoreBeautifulPython\build_and_push.py
    found examples.py <== C:\\Users\sudon\MoreBeautifulPython\examples.py
    found mbp.py <== C:\\Users\sudon\MoreBeautifulPython\src\mbp.py
    found __init__.py <== C:\\Users\sudon\MoreBeautifulPython\src\__init__.py
    ==================================================================================
    '''

    # make_files() and make_dirs() create files and directory after creating paths
    with enclose('make_files() and make_dirs()'):
        test_dir = './test_dir'
        make_files([join_path(test_dir, file_name) for file_name in ['a.txt', 'b.txt', 'c.txt']])
        log('{} files under {}'.format(len(list(open_files(test_dir))), test_dir))
        make_dirs(test_dir, overwrite=True)
        log('{} files under {} (after overwrite)'.format(len(list(open_files(test_dir))), test_dir))
        shutil.rmtree(test_dir)
    '''
    ======= make_files() and make_dirs() =======
    found a.txt <== ./test_dir\a.txt
    found b.txt <== ./test_dir\b.txt
    found c.txt <== ./test_dir\c.txt
    3 files under ./test_dir
    0 files under ./test_dir (after overwrite)
    ============================================
    '''

    # type_in() return the type idx of 1st argument defined by the 2nd argument
    with enclose('type_in()'):
        types = [int, dict, list, set]
        # the return idx started at 1 because 0 is reserved for no match
        idx = type_in([1, 2, 3], types) - 1
        log(types[idx])
        if not type_in("a string", types):
            log('this type is not from the list')
    '''
    ========== type_in() ===========
    <class 'list'>
    this type is not from the list
    ================================
    '''

    # get_range() is a replacement for range(len())
    with enclose('get_range() and get_items()'):
        vec = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        vec_iter = iter(vec)
        log(list(get_range(vec)))
        log(list(get_range(vec, 2, 4)))
        log(list(get_items(vec, 2, None, 2, reverse=True)))
        log(list(get_items(vec_iter, 0, 5)))
    '''
    ===== get_range() and get_items() =====
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    [2, 3]
    ['i', 'g', 'e', 'c']
    ['a', 'b', 'c', 'd', 'e']
    =======================================
    '''

    # prints() is a superior pprint()
    with enclose("prints()"):
        multi_line = "line1\n*   line2\n*   line3\n"
        long_strings = ['-' * 60] * 3
        long_list = [i for i in range(70)]
        nested_list = [[[1, 2, 3], [4, 5, 6]]]
        hybrid_list = [{'abc', 'bcd'}] + long_list + [(1, 2, 3)] + [[[multi_line, multi_line]]] + [0, 1, 2]
        hybrid_dict = {'a': hybrid_list, '': 'empty_key',
                       'c': {'d': nested_list, 'e': long_strings, 'f': {'a longer key': long_list}}}
        prints(hybrid_dict)
    '''
    ================================================== prints() ===================================================
    {
        "a": [{"abc","bcd"},
              0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
              30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,
              56,57,58,59,60,61,62,63,64,65,66,67,68,69,
              (1,2,3),
              [["line1\n"
                "*   line2\n"
                "*   line3\n",
                "line1\n"
                "*   line2\n"
                "*   line3\n"]],
              0,1,2],
        "": "empty_key",
        "c": {
            "d": [[[1,2,3],
                   [4,5,6]]],
            "e": ["------------------------------------------------------------",
                  "------------------------------------------------------------",
                  "------------------------------------------------------------"],
            "f": {
                "a longer key": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
                                 30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,
                                 56,57,58,59,60,61,62,63,64,65,66,67,68,69]
            }
        }
    }
    ===============================================================================================================
    '''

    # try_f() perform try-except routine and capture the result or error messages in a dictionary
    prints(try_f(test_sleep_and_fail, 'input', fail_rate=1))
    '''
    {
        "error": "AssertionError('simulated failure (100%)')",
        "error_type": "AssertionError",
        "error_msg": "simulated failure (100%)",
        "traceback": "Traceback (most recent call last):\n"
                     "  File "C:\\Users\sudon\MoreBeautifulPython\src\mbp.py", line 670, in try_f\n"
                     "    res['res'] = f(*args[1:], **kwargs)\n"
                     "  File "C:\\Users\sudon\MoreBeautifulPython\examples.py", line 14, in test_sleep_and_fail\n"
                     "    assert random.random() > fail_rate, "simulated failure ({}%)".format(fail_rate * 100)\n"
                     "AssertionError: simulated failure (100%)\n"
    }
    '''
    prints(try_f(test_sleep_and_fail, 'input', fail_rate=0))
    '''
    {
        "res": "input"
    }
    '''

    # check() can trace back to the original string of the function call and print the variable names and values
    # check() is slow and should be used only for inspection purposes.
    a = 123
    check(a, multi_line, nested_list)  # your comment
    '''
    ----- [main]: check(a, multi_line, nested_list)  # your comment -----
    a = 123
    multi_line = "line1\n"
                 "*   line2\n"
                 "*   line3\n"
    nested_list = [[[1,2,3],
                    [4,5,6]]]
    ---------------------------------------------------------------------
    '''

    # debug() == check(level=DEBUG)
    debug(multi_line, nested_list)

    # join_path() == os.path.join()
    jsonl_file_path = join_path(this_dir(), 'data.jsonl')
    # load_jsonl() return an iterator of dictionary
    data = list(load_jsonl(join_path(this_dir(), 'data.jsonl')))
    # print_iter(iterator) == [log(item) for item in iterator]
    with enclose('print_list()'):
        print_iter(data)
    '''
    ============= print_list() ==============
    {'id': 1, 'name': 'Jackson', 'age': 43}
    {'id': 2, 'name': 'Zunaira', 'age': 24}
    {'id': 3, 'name': 'Lorelei', 'age': 72}
    =========================================
    '''

    # print_table() can adjust column width automatically
    with enclose('print_table()'):
        rows = [list(d.values()) for d in data]
        column_names = list(data[0].keys())
        print_table([column_names] + rows, space=3)
    '''
    ===== print_table() =====
    id   name      age
    1    Jackson   43
    2    Zunaira   24
    3    Lorelei   72
    =========================
    '''

    # print_table() can also handle tables inside table
    with enclose('tables inside table'):
        rows = [['total', 140], ['stats', [['a: ', 70], ['b: ', 20], ['c: ', 50]]]]
        print_table(rows)
    '''
    ===== tables inside table =====
    total   140
    stats   a:  70
            b:  20
            c:  50
    ===============================
    '''

    # get 3 key statistics from an iterator at once
    log(n_min_max_avg(load_jsonl(jsonl_file_path), key_f=lambda x: x['age']))
    log(min_max_avg(load_jsonl(jsonl_file_path), key_f=lambda x: x['age']))
    log(avg(load_jsonl(jsonl_file_path), key_f=lambda x: x['age']))
    '''
    (3, 24, 72, 46.333333333333336)
    (24, 72, 46.333333333333336)
    46.333333333333336
    '''

    # curr_time() == str(datetime.now(timezone.utc))[:19]
    log(curr_time())
    '''
    2022-09-10 20:53:53
    '''


if __name__ == '__main__':
    # main()
    prints({'a': []})