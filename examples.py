import sys
import shutil
import time
import random
# from mbp import * (after pip install)
from src.mbp import *
from src.mbp.llm import *


# test functions for multiprocessing
def sleep_then_maybe_fail(x, duration=0.2, fail_p=0):
    time.sleep(duration)
    assert random.random() > fail_p, "simulated failure ({}%)".format(fail_p * 100)
    return x


def read_from_vec(idx, vec):
    return vec[idx]


def build_vec(size):
    return [x for x in range(size)]


def fn(*functions):
    return ', '.join(f.__name__ + '()' for f in functions)


def alog(res, reference):
    log(res)
    assert res == reference, f"{res} != {reference}"


def test_core(log_path='./examples.log'):
    # log() include all functionality of print()
    log('this is from the global logger')

    # from this point on, all log() will print to file at path "./log" as well as stdout
    set_global_logger(file=[log_path, sys.stdout])

    # local_logger() return a local logger just like logging.getLogger
    my_logger = local_logger(__name__, verbose=True)
    my_logger('this is from the local logger', level=WARNING)

    # context_logger() (as context manager) temporarily modify the global logger
    with context_logger(level=DEBUG, file=sys.stderr, name='__temp__', verbose=True):
        # this message will be redirected to sys.stderr
        log('this is from the temporary logger (level={})'.format(
            global_logger_level()), level=CRITICAL)

    # suppress all logs by setting level=SILENT
    with context_logger(level=SILENT):
        for i in range(1000):
            # all loggings are suppressed
            log(i)

    # context_logger() will also overwrite all logger() within its scope
    with context_logger(level=INFO):
        with context_logger(level=SILENT):
            log('==> a hidden message')

    # except when can_overwrite == False
    with context_logger(level=INFO):
        with context_logger(level=SILENT, can_overwrite=False):
            log('==> this will never be printed')

    # print_line() will draw a line with am optional message
    print_line()
    print_line(text='optional message', width=30, char='=')
    log('\n')

    # enclose() generate two text separators that enclose the execution
    with enclose('header', width=30, bottom_margin=1, char='=', use_timer=False):
        [log('this is line {}'.format(i)) for i in range(3)]

    # when width=None, the width will be calculated automatically based on the captured content
    with enclose():
        [log('this is line {}'.format(i)) for i in range(3)]

    # recorder() save all logs into a (passed-in) list
    with enclose(fn(recorder)):
        tape = []
        with recorder(tape, captured_level=DEBUG):
            log('9 8 7 6 5 4 3 2 1')
            log('ok')
            log('a debug message', level=DEBUG)
        tape[0] = tape[0][::-1]
        print_iter(tape)

    # timer() as a context manager
    with timer('build_vec(100000)'):
        build_vec(100000)

    # enclose_timer() == enclose(timer=True)
    # iterate() can customize iteration procedures
    # for example, sample 10% and report every 3 yield from the first 100 samples
    with enclose_timer():
        for d in iterate(range(1000), first_n=100, sample_p=0.05, report_n=2):
            log(sleep_then_maybe_fail(d, duration=0.1))

    # Workers() is more flexible than multiprocessing.Pool()
    n_task = 6
    with enclose_timer(fn(Workers)):
        workers = Workers(f=sleep_then_maybe_fail, num_workers=3, verbose=True, ignore_error=True)
        [workers.add_task({'x': i, 'fail_p': 0.3}) for _ in range(n_task)]
        [workers.get_result() for _ in range(n_task)]
        workers.terminate()

    # similarly, we can use work() to process tasks from an iterator
    # tasks can be iterator of tuple (need to specify all inputs) or dict
    with enclose_timer(fn(work)):
        tasks = iter([(i, 0.2, 0.5) for i in range(n_task)])
        for r in work(f=sleep_then_maybe_fail, tasks=tasks, ordered=True, ignore_error=True):
            log(r)

    # use cached_inp = {'fixed_input': value, ...} to avoid pickling of heavy objects
    with enclose(fn(work) + " with cache_inp"):
        vec_size = 1000000
        vec = [x for x in range(vec_size)]
        with timer('work()'):
            a = list(work(read_from_vec, num_workers=1, tasks=iter((i, vec) for i in range(30))))
        with timer('work() with cache_inp'):
            b = list(work(read_from_vec, num_workers=1, tasks=iter({'idx': i} for i in range(30)), cache_inp={'vec': vec}))
        assert a == b

        # for objects that can not be pickled, use built_inp
        with timer('work() with build_inp'):
            tasks = iter({'idx': i} for i in range(30))
            list(work(read_from_vec, num_workers=1, tasks=tasks, build_inp={'vec': (build_vec, vec_size)}))

    # jpath() == os.path.join()
    # run_dir() == os.getcwd()
    # this_dir() return the directory of the current file
    # dir_basename() check and return the directory name of a path
    # file_basename() check and return the file name of a path
    with enclose(fn(jpath, run_dir, lib_path, this_dir, dir_basename, file_basename)):
        log(jpath(this_dir(), 'a', 'b', 'c.file'))
        log(run_dir())
        log(lib_path())
        log(this_dir())
        log(this_dir(go_up=1, go_to='AnotherProject/hello.txt'))
        log(dir_basename(dir_of(__file__)))
        log(file_basename(__file__))

    # open_files() return all files under a directory
    with enclose(fn(open_files)):
        for f in open_files(this_dir(), pattern=r'.*\.py'):
            f.readlines()

    # build_files() (or build_dirs()) create files (or directory) after creating paths
    with enclose(fn(build_files, build_dirs)):
        test_dir = './test_dir'
        build_files([jpath(test_dir, file_name) for file_name in ['a.txt', 'b.txt', 'c.txt']])
        log('{} files under {}'.format(len(list(open_files(test_dir))), test_dir))
        build_dirs(test_dir, overwrite=True)
        log('{} files under {} (after overwrite)'.format(len(list(open_files(test_dir))), test_dir))
        shutil.rmtree(test_dir)

    # unwrap_file() (or unwrap_dir()) will unwrap all parent directories leading to a unique file (or dir)
    with enclose(fn(unwrap_file, unwrap_dir)):
        build_files('./a/b/c/file')
        log(unwrap_file('./a'))
        log(unwrap_dir('./a'))
        shutil.rmtree('./a')

    # type_of() return the type idx of 1st argument defined by the 2nd argument
    with enclose(fn(type_of)):
        types = [int, dict, list, set]
        # the return idx started at 1 because 0 is reserved for no match
        idx = type_of([1, 2, 3], types) - 1
        log(types[idx])
        if not type_of("a string", types):
            log('this type is not from the list')

    # for i in range_of(data)              ==   for i in range(len(data))
    # for d in items_of(data, 1, 5, 2)     ==   for d in itertools.islice(data, start=1, end=, step=2)
    with enclose(fn(range_of, items_of)):
        vec = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        vec_iter = iter(vec)
        log(list(range_of(vec)))
        log(list(range_of(vec, 2, 4)))
        log(list(items_of(vec, 2, None, 2, reverse=True)))
        log(list(items_of(vec_iter, 0, 5)))

    # prints() is a superior pprint()
    with enclose(fn(prints)):
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

    # try_f() perform try-except routine and capture the result or error messages in a dictionary
    with enclose(fn(try_f)):
        prints(try_f(sleep_then_maybe_fail, 'abc', fail_p=1))
        log(try_f(sleep_then_maybe_fail, 'abc', fail_p=0))

    # break_str() break a long string into list of smaller (measured by wcswidth()) strings
    with enclose(fn(break_str)):
        string = 'a very very very very long string'
        alog(break_str(string, width=12), ["a very very ", "very very lo", "ng string"])

    # shorten_str() truncate a string and append "..." if len(string) > width
    with enclose(fn(shorten_str)):
        alog(shorten_str(string, 100), "a very very very very long string")
        alog(shorten_str(string, 20), "a very very very ...")

    # fill_str() replace placeholders in string with an arguments
    with enclose(fn(fill_str)):
        alog(fill_str("1{ok}34{ok2}", ok=2, ok2=5), "12345")

    # debug() can trace back to the original function call and print the variable names with their values
    # debug() is slow and should be used only for inspection purposes.
    a = 123
    b = [1, 2, 3, 4]
    c = 'line1\nline2\nline3'
    # nothing happen because the current LOGGER.level = INFO
    debug(a)

    # this will print to the console but not to ./log
    with context_logger(level=DEBUG):
        debug(a, b, c)
        debug(a, b, c, mode=prints)
        debug(b, mode=print_iter)

    # load_jsonl() return an iterator of dictionary
    jsonl_file_path = jpath(this_dir(), 'data.jsonl')
    data = list(load_jsonl(jsonl_file_path))

    # print_iter(iterator) == [log(item) for item in iterator]
    with enclose(fn(print_iter)):
        print_iter(data)

    # print_table() can adjust column width automatically
    rows = [list(d.values()) for d in data]
    headers = list(data[0].keys())
    print_table(rows, name=fn(print_table), headers=headers, space=3)

    # print_table() can also pad a row, and handle tables inside table (if item is a list, dict, set, or tuple)
    # print_table() calculate column width based on the longest item or use min_column_widths if applicable
    rows += [[6, 'Paimon'], ['', 'summary',
                             [['num characters', 6], ['num cities', 4]]]]
    print_table(rows,
                headers=headers,
                min_column_widths=[None, 20],
                name=fn(print_table) + ' with incomplete rows')

    # use max_column_width to shorten a cell with long data (str)
    print_table([[1, 2, '3' * 100], [1, '2' * 100, 3]],
                headers=['a', 'b', 'c'],
                max_column_width=10,
                name=fn(print_table) + " with long cell")

    # get 3 key statistics from an iterator at once
    with enclose(fn(n_min_max_avg, min_max_avg, avg)):
        numbers = [1, 2, 3, 4, 5]
        alog(n_min_max_avg(numbers), (5, 1, 5, 3.0))
        alog(min_max_avg(numbers), (1, 5, 3.0))
        alog(avg(numbers), 3.0)

    # curr_time() == str(datetime.now(timezone.utc))[:19]
    with enclose(fn(curr_time)):
        log(curr_time())
        log(curr_time(breakdown=True))


def test_llm():
    log(build_system_message(
        "Let's think about this math problem step by step",
        outputs=["res"],
        examples=[
            {"input": "5 + 5", "res": 10},
            {"input": "2 * (3 + 2)", "res": 10}
        ]))


if __name__ == '__main__':
    test_core()
    test_llm()
