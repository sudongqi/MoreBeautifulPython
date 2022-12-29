import os
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


def build_vec(size):
    return [x for x in range(size)]


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
    print_line(30)
    print_line(30, text='optional message', char='=')
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
    ==================================== work() ====================================
    {'worker_id': 1, 'task_id': 0, 'res': 0}
    {'worker_id': 2, 'task_id': 1, 'res': None, 'error': "AssertionError('simulated failure (50.0%)')"}
    {'worker_id': 0, 'task_id': 2, 'res': None, 'error': "AssertionError('simulated failure (50.0%)')"}
    {'worker_id': 3, 'task_id': 3, 'res': 3}
    {'worker_id': 5, 'task_id': 4, 'res': None, 'error': "AssertionError('simulated failure (50.0%)')"}
    {'worker_id': 4, 'task_id': 5, 'res': 5}
    ================================================================================
    took 345.680 ms
    '''

    with enclose("work() with cache_inp"):
        # use cached_inp = {'fixed_input': value, ...} to avoid pickling of heavy objects
        vec_size = 1000000
        vec = [x for x in range(vec_size)]
        with timer('work()'):
            a = list(work(test_read_from_list, num_workers=1, ordered=True, tasks=iter((i, vec) for i in range(30))))
        with timer('work() with cache_inp'):
            b = list(work(test_read_from_list, num_workers=1, ordered=True, tasks=iter({'idx': i} for i in range(30)),
                          cache_inp={'vec': vec}))
        assert a == b

        # for objects that can not be pickled, use built_inp
        with timer('work() with build_inp'):
            tasks = iter({'idx': i} for i in range(30))
            list(work(test_read_from_list, num_workers=1, ordered=True, tasks=tasks,
                      build_inp={'vec': (build_vec, vec_size)}))
    '''
    ========== work() with cache_inp ==========
    work() ==> took 1890.690 ms
    work() with cache_inp ==> took 109.339 ms
    work() with build_inp ==> took 109.322 ms
    ===========================================
    '''

    with enclose('pathing'):
        # jpath() == os.path.join()
        log(jpath(this_dir(), 'a', 'b', 'c.file'))
        # exec_dir() == os.getcwd()
        log(run_dir())
        # lib_path() return the path of the mbp library
        log(lib_path())
        # this_dir() return the directory of the current file
        log(this_dir())
        log(this_dir(go_up=1, go_to='AnotherProject/hello.txt'))
        # get_dir_name() check and return the directory name of a path
        log(dir_basename(dir_of(__file__)))
        # get_file_name() check and return the file name of a path
        log(file_basename(__file__))
    '''
    =================== pathing ===================
    C:/Users/sudon/MoreBeautifulPython/a/b/c.file
    C:/Users/sudon/MoreBeautifulPython
    C:/Users/sudon/MoreBeautifulPython/src/mbp.py
    C:/Users/sudon/MoreBeautifulPython
    C:/Users/sudon/AnotherProject/hello.txt
    MoreBeautifulPython
    examples.py
    ===============================================
    '''

    # open_files() return all files under a directory
    with enclose('open_files()'):
        for f in open_files(this_dir(), pattern=r'.*\.py'):
            f.readlines()
    '''
    ============================== open_files() ==============================
    found examples.py <== C:/Users/sudon/MoreBeautifulPython/examples.py
    found sync.py <== C:/Users/sudon/MoreBeautifulPython/sync.py
    found mbp.py <== C:/Users/sudon/MoreBeautifulPython/src/mbp.py
    found __init__.py <== C:/Users/sudon/MoreBeautifulPython/src/__init__.py
    ==========================================================================
    '''

    # init_files() and init_dirs() create files and directory after creating paths
    with enclose('init_files() and init_dirs()'):
        test_dir = './test_dir'
        init_files([jpath(test_dir, file_name) for file_name in ['a.txt', 'b.txt', 'c.txt']])
        log('{} files under {}'.format(len(list(open_files(test_dir))), test_dir))
        init_dirs(test_dir, overwrite=True)
        log('{} files under {} (after overwrite)'.format(len(list(open_files(test_dir))), test_dir))
        shutil.rmtree(test_dir)
    '''
    ================== init_files() and init_dirs() ===================
    found a.txt <== C:/Users/sudon/MoreBeautifulPython/test_dir/a.txt
    found b.txt <== C:/Users/sudon/MoreBeautifulPython/test_dir/b.txt
    found c.txt <== C:/Users/sudon/MoreBeautifulPython/test_dir/c.txt
    3 files under ./test_dir
    0 files under ./test_dir (after overwrite)
    ===================================================================
    '''

    # unwrap_file()/unwrap_dir() will unwrap all parent directories leading to a unique file/dir
    with enclose('unwrap_path() and unwrap_dir()'):
        init_files('./a/b/c/file')
        log(unwrap_file('./a'))
        log(unwrap_dir('./a'))
        shutil.rmtree('./a')
    '''
    ===== unwrap_path() and unwrap_dir() =====
    ./a/b/c/file
    ./a/b/c
    ==========================================
    '''

    # type_in() return the type idx of 1st argument defined by the 2nd argument
    with enclose('type_in()'):
        types = [int, dict, list, set]
        # the return idx started at 1 because 0 is reserved for no match
        idx = type_of([1, 2, 3], types) - 1
        log(types[idx])
        if not type_of("a string", types):
            log('this type is not from the list')
    '''
    ========== type_in() ===========
    <class 'list'>
    this type is not from the list
    ================================
    '''

    # for i in range_of(data)              ==   for i in range(len(data))
    # for d in items_of(data, 1, 5, 2)     ==   for d in itertools.islice(data, start=1, end=, step=2)
    with enclose('range_of() and items_of()'):
        vec = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        vec_iter = iter(vec)
        log(list(range_of(vec)))
        log(list(range_of(vec, 2, 4)))
        log(list(items_of(vec, 2, None, 2, reverse=True)))
        log(list(items_of(vec_iter, 0, 5)))
    '''
    ===== range_of() and items_of() =====
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    [2, 3]
    ['i', 'g', 'e', 'c']
    ['a', 'b', 'c', 'd', 'e']
    =======================================
    '''

    # prints() is a superior pprint()
    with enclose("prints()"):
        class TestObject:
            def __str__(self):
                return 'this is a test object'

        _tuple = (1, 2, 3, None)
        _set = {'abc', 'bcd'}
        _object = TestObject()
        list_of_long_strings = ['-' * 60] * 3
        short_list = ['a', 'b', 'c', 'd']
        long_list = [i for i in range(70)]
        nested_list = [[[1, 2, 3], [4, 5, 6]]]
        multi_lines = "line1\n\t\t- line2\n\t\t- line3\n"
        hybrid_list = [_set, _tuple, long_list, short_list, {}, [], (),
                       "string", None, True, False, (1, 2, 3), nested_list, multi_lines]
        hybrid_dict = {'hybrid_dict': hybrid_list, '': 'empty key', 2048: 'number key', 'object': _object,
                       'nested_dict': {'a': nested_list, 'b': list_of_long_strings,
                                       'c': {'a longer key': long_list}}}
        prints(hybrid_dict)
    '''
    =================================== prints() ===================================
    {
        "hybrid_dict": [{"abc","bcd"},
                        (1,2,3,None),
                        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
                         30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,
                         56,57,58,59,60,61,62,63,64,65,66,67,68,69],
                        ["a","b","c","d"],
                        {},
                        [],
                        (),
                        "string",None,True,False,
                        (1,2,3),
                        [[[1,2,3],
                          [4,5,6]]],
                        "line1\n"
                        "           - line2\n"
                        "           - line3\n"],
        "": "empty key",
        2048: "number key",
        "object": "this is a test object",
        "nested_dict": {
            "a": [[[1,2,3],
                   [4,5,6]]],
            "b": ["------------------------------------------------------------",
                  "------------------------------------------------------------",
                  "------------------------------------------------------------"],
            "c": {
                "a longer key": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
                                 30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,
                                 56,57,58,59,60,61,62,63,64,65,66,67,68,69]
            }
        }
    }
    ================================================================================
    '''

    # try_f() perform try-except routine and capture the result or error messages in a dictionary
    prints(try_f(test_sleep_and_fail, 'input', fail_rate=1))
    '''
    {
        "error": "AssertionError('simulated failure (100%)')",
        "error_type": "AssertionError",
        "error_msg": "simulated failure (100%)",
        "traceback": "Traceback (most recent call last):\n"
                     "  File "C:/Users/sudon/MoreBeautifulPython/src/mbp.py", line 758, in try_f\n"
                     "    res['res'] = f(*args[1:], **kwargs)\n"
                     "  File "C:/Users/sudon/MoreBeautifulPython/examples.py", line 13, in test_sleep_and_fail\n"
                     "    assert random.random() > fail_rate, "simulated failure ({}%)".format(fail_rate * 100)\n"
                     "AssertionError: simulated failure (100%)\n"
    }
    '''
    log(try_f(test_sleep_and_fail, 'input', fail_rate=0))
    '''
    {'res': 'input'}
    '''

    # debug() can trace back to the original function call and print the variable names with their values
    # debug() is slow and should be used only for inspection purposes.
    a = 123
    b = [1, 2, 3, 4]
    c = 'line1\nline2\nline3'

    # nothing happen because the current LOGGER.level = INFO
    debug(a)

    # this will print to the console but not to ./log
    with logger(level=DEBUG):
        debug(a, b, c)
        debug(a, b, c, mode=prints)
        debug(b, mode=print_iter)
    '''
    ----- [394] examples.py <main> -----
    a:  123
    b:  [1, 2, 3, 4]
    c:  line1
        line2
        line3
    ------------------------------------
    
    ----- [395] examples.py <main> -----
    a: 123
    b: [1,2,3,4]
    c: "line1\n"
       "line2\n"
       "line3"
    ------------------------------------
    
    ----- [396] examples.py <main>: b -----
    1
    2
    3
    4
    ---------------------------------------
    '''

    # load_jsonl() return an iterator of dictionary
    jsonl_file_path = jpath(this_dir(), 'data.jsonl')
    data = list(load_jsonl(jsonl_file_path))

    # print_iter(iterator) == [log(item) for item in iterator]
    with enclose('print_list()'):
        print_iter(data)
    '''
    ================= print_list() =================
    {'id': 1, 'name': 'Jean', 'city': 'Mondstadt'}
    {'id': 2, 'name': 'Xingqiu', 'city': 'Liyue'}
    {'id': 3, 'name': 'Ganyu', 'city': 'Liyue'}
    {'id': 4, 'name': 'Ayaka', 'city': 'Inazuma'}
    {'id': 5, 'name': 'Nilou', 'city': 'Sumeru'}
    ================================================
    '''

    # print_table() can adjust column width automatically
    rows = [list(d.values()) for d in data]
    headers = list(data[0].keys())
    print_table(rows, headers=headers, space=3)
    '''
    -------------------
    id   name      city
    -------------------
    1    Jean      Mondstadt
    2    Xingqiu   Liyue
    3    Ganyu     Liyue
    4    Ayaka     Inazuma
    5    Nilou     Sumeru
    -------------------
    '''

    # print_table() can also pad a row, and handle tables inside table (if item is a list, dict, set, or tuple)
    rows += [[6, 'Paimon'], ['', 'summary', [['num characters', 6], ['num cities', 4]]]]
    print_table(rows, headers)
    '''
    -------------------
    1    Jean      Mondstadt
    2    Xingqiu   Liyue
    3    Ganyu     Liyue
    4    Ayaka     Inazuma
    5    Nilou     Sumeru
    6    Paimon
         summary   num characters 6
                   num cities     4
    -------------------
    '''

    # break_string() break a long string into list of smaller (measured by wcswidth()) strings
    log(break_string('a' * 20, width=5))

    numbers = [1, 2, 3, 4, 5]
    # get 3 key statistics from an iterator at once
    log(n_min_max_avg(numbers))
    log(min_max_avg(numbers))
    log(avg(numbers))
    '''
    (5, 1, 5, 3.0)
    (1, 5, 3.0)
    3.0
    '''

    # curr_time() == str(datetime.now(timezone.utc))[:19]
    log(curr_time())
    log(curr_time(breakdown=True))
    '''
    2022-09-10 20:53:53
    ('2022', '11', '06', '23', '24', '00')
    '''


if __name__ == '__main__':
    main()