import sys
import shutil
import time
import random
import asyncio

# from mbp import * (for normal usage)
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


def fname(*functions):
    return ", ".join(f.__name__ + "()" for f in functions)


def log_assert(res, reference):
    log(res)
    if callable(reference):
        assert reference(res)
    else:
        assert res == reference, f"{res} != {reference}"


async def test_core(log_path="./test.log"):
    # log() include all functionality of print()
    log("this is from the global logger", end="\n\n", color="blue")

    # from this point on, all log() will print to file at path "./log" as well as stdout
    set_logger(file=[log_path, sys.stdout])

    # local_logger() return a local logger just like logging.getLogger
    local_logger = logger(__name__, verbose=True)
    local_logger("this is from the local logger", level=WARNING)

    # context_logger() (as context manager) temporarily modify the global logger
    with block_logger(level=DEBUG, file=sys.stderr, name="__temp__", verbose=True):
        # this message will be redirected to sys.stderr
        log(f"this is from the temporary logger (level={get_logger().level})", level=CRITICAL)

    # suppress all logs by setting level=ERROR
    with block_logger(level=ERROR):
        for i in range(1000):
            # all loggings are suppressed
            log(i)

    # context_logger() will also overwrite all logger() within its scope
    with block_logger(level=INFO):
        with block_logger(level=WARNING):
            log("==> a hidden message")

    # except when can_overwrite == False
    with block_logger(level=INFO):
        with block_logger(level=WARNING, can_overwrite=False):
            log("==> this will never be printed")

    # print_line() will draw a line with am optional message
    print_line()
    print_line("line")
    print_line(40)
    print_line("line", 40)
    print_line("line", width=40, color="light_red")
    print_line(text="line", width=40, char="=")
    log("\n")

    # block() generate two text separators that enclose the execution
    with block("header", width=30, char="=", color="blue"):
        [log("this is line {}".format(i), color="red") for i in range(3)]

    # when width=None, the width will be calculated automatically based on the captured content
    with block():
        [log("this is line {}".format(i)) for i in range(3)]

    # recorder() save all logs into a (passed-in) list
    with block(fname(recorder)):
        with recorder(captured_level=DEBUG) as r:
            log("9 8 7 6 5 4 3 2 1")
            log("ok")
            log("a debug message", level=DEBUG)
        recording = r.flush()
        assert recording.startswith("9 8 7") and recording.endswith("debug message\n")
        log(recording, end="")

    # timer() as a context manager
    with timer("build_vec(100000)"):
        build_vec(100000)

    # iterate() can customize iteration procedures
    # for example, sample 10% and report every 3 yield from the first 100 samples
    with block_timer():
        for d in iterate(range(1000), first_n=100, sample_p=0.05, report_n=2):
            log(sleep_then_maybe_fail(d, duration=0.1))

    # Workers() is more flexible than multiprocessing.Pool()
    n_task = 6
    with block_timer(fname(Workers)):
        workers = Workers(f=sleep_then_maybe_fail, num_workers=3, verbose=True, ignore_error=True)
        [workers.add_task({"x": i, "fail_p": 0.3}) for _ in range(n_task)]
        [workers.get_result() for _ in range(n_task)]
        workers.terminate()

    # similarly, we can use work() to process tasks from an iterator
    # tasks can be iterator of tuple (need to specify all inputs) or dict
    # r is None ==> task failed
    with block_timer(fname(work)):
        tasks = iter([(i, 0.2, 0.5) for i in range(n_task)])
        for r in work(f=sleep_then_maybe_fail, tasks=tasks, ordered=True, ignore_error=True):
            log(r)

    # use cached_inp = {'fixed_input': value, ...} to avoid pickling of heavy objects
    with block(fname(work) + " with cache_inp"):
        vec_size = 100000
        vec = [x for x in range(vec_size)]
        with timer("work()"):
            a = list(work(read_from_vec, num_workers=1, tasks=iter((i, vec) for i in range(30))))
        with timer("work() with cache_inp"):
            tasks = iter({"idx": i} for i in range(30))
            b = list(work(read_from_vec, num_workers=1, tasks=tasks, cache_inp={"vec": vec}))
        assert a == b

        # for objects that can not be pickled, use built_inp
        with timer("work() with build_inp"):
            tasks = iter({"idx": i} for i in range(30))
            list(work(read_from_vec, num_workers=1, tasks=tasks, build_inp={"vec": (build_vec, vec_size)}))

    # jpath() == os.path.join()
    # run_dir() == os.getcwd()
    # this_dir() return the directory of the current file
    # dir_basename() check and return the directory name of a path
    # file_basename() check and return the file name of a path
    with block(fname(jpath, run_dir, lib_path, this_dir, dir_basename, file_basename)):
        log_assert(jpath("/my_folder", "a", "b", "c.file"), "/my_folder/a/b/c.file")
        log(run_dir())
        log(lib_path())
        log_assert(this_dir(), lambda x: x.endswith("MoreBeautifulPython"))
        log(this_dir(up=1, to="AnotherProject/hello.txt"))
        log_assert(dir_basename(dir_of(__file__)), "MoreBeautifulPython")
        log_assert(file_basename(__file__), "run.py")

    # scan_path is a wrapper of os.scandir
    with block(fname(scan_path)):
        num_files = len(list(scan_path(this_dir())))
        num_folders = len(list(scan_path(this_dir(), include_dirs=True, include_files=False)))
        num_total = len(list(scan_path(this_dir(), include_dirs=True)))
        num_files_exclude_py = len(list(t[1] for t in scan_path(this_dir(), ignore=["*.py"])))
        prints({"num_files": num_files, "num_files_exclude_py": num_files_exclude_py, "num_folders": num_folders, "num_total": num_total})
        assert num_files + num_folders == num_total
        assert num_files_exclude_py < num_files

    # build_files() (or build_dirs()) create files (or directory) after creating paths
    with block(fname(build_files, build_dirs)):
        test_dir = "./test_dir"
        build_files([jpath(test_dir, file_name) for file_name in ["a.txt", "b.txt", "c.txt"]])
        assert len(list(scan_path(test_dir))) == 3
        build_dirs(test_dir, overwrite=True)
        assert len(list(scan_path(test_dir))) == 0
        shutil.rmtree(test_dir)

    # unwrap_file() (or unwrap_dir()) will unwrap all parent directories leading to a unique file (or dir)
    with block(fname(unwrap_file, unwrap_dir)):
        build_files("./a/b/c/file")
        log(unwrap_file("./a"))
        log(unwrap_dir("./a"))
        shutil.rmtree("./a")

    # type_of() return the type idx of 1st argument defined by the 2nd argument
    with block(fname(type_of)):
        types = [int, dict, list, set]
        # the return idx started at 1 because 0 is reserved for no match
        idx = type_of([1, 2, 3], types) - 1
        log(types[idx])
        if not type_of("a string", types):
            log("this type is not from the list")

    # prints() is a superior pprint()
    with block(fname(prints)):

        class TestObject:
            def __str__(self):
                return "this is a test object"

        _tuple = (1, 2, 3, None)
        _set = {"abc", "bcd"}
        _object = TestObject()
        list_of_long_strings = ["-" * 60] * 3
        short_list = ["a", "b", "c", "d"]
        long_list = [i for i in range(70)]
        nested_list = [[[1, 2, 3], [4, 5, 6]]]
        multi_lines = "line1\n\t\t- line2\n\t\t- line3\n"
        hybrid_list = [_set, _tuple, long_list, short_list, {}, [], (), "string", None, True, False, (1, 2, 3), nested_list, multi_lines]
        hybrid_dict = {
            "hybrid_dict": hybrid_list,
            "": "empty key",
            2048: "number key",
            "object": _object,
            "nested_dict": {"a": nested_list, "b": list_of_long_strings, "c": {"a longer key": long_list}},
        }
        prints(hybrid_dict)

    # merge() is a recursive dict.update()
    a = {"a": {"a": 100, "b": [0]}}
    b = {"a": {"b": 201, "c": 301}}
    c = {"a": {"c": 303}, "c": [1, 2]}
    log_assert(dmerge(a, b, c), {"a": {"a": 100, "b": 201, "c": 303}, "c": [1, 2]})
    log_assert(dmerge(c, b, a), {"a": {"a": 100, "b": [0], "c": 301}, "c": [1, 2]})

    # try_f() perform try-except routine and capture the result or error messages in a dictionary
    with block(fname(try_f)):
        prints(try_f(sleep_then_maybe_fail, "abc", fail_p=1), color="magenta")
        log(try_f(sleep_then_maybe_fail, "abc", fail_p=0))

    # break_str() break a long string into list of smaller (measured by wcswidth()) strings
    with block(fname(break_str)):
        string = "a very very very very long string"
        log_assert(break_str(string, width=12), ["a very very ", "very very lo", "ng string"])

    # shorten_str() truncate a string and append "..." if len(string) > width
    with block(fname(shorten_str)):
        log_assert(shorten_str(string, 100), "a very very very very long string")
        log_assert(shorten_str(string, 20), "a very very very ...")

    # fill_str() replace placeholders in string with an arguments
    with block(fname(fill_str)):
        log_assert(fill_str("1{ok}34{ok2}", ok=2, ok2=5), "12345")

    # debug() can trace back to the original function call and print the variable names with their values
    # debug() is slow and should be used only for inspection purposes.
    a = 123
    b = [1, 2, 3, 4]
    c = "line1\nline2\nline3"
    # nothing happen because the current LOGGER.level = INFO
    debug(a)

    # this will print to the console but not to ./log
    with block_logger(level=DEBUG):
        debug(a, b, c)
        debug(a, b, c, mode=prints)
        debug(b, mode=print_iter)

    # load_jsonl() return an iterator of dictionary
    jsonl_file_path = jpath(this_dir(), "data/test.jsonl")
    data = list(iter_jsonl(jsonl_file_path))

    # print_iter(iterator) == [log(item) for item in iterator]
    with block(fname(print_iter)):
        print_iter(data)

    # print_table() can adjust column width automatically
    rows = [list(d.values()) for d in data]
    headers = list(data[0].keys())
    print_table(rows, name=fname(print_table), headers=headers, space=3)
    print()

    # print_table() can also pad a row, and handle tables inside table (if item is a list, dict, set, or tuple)
    # print_table() calculate column width based on the longest item or use min_column_widths if applicable
    rows += [[6, "Paimon"], ["", "summary", [["num characters", 6], ["num cities", 4]]]]
    print_table(rows, headers=headers, min_column_widths=[None, 20], name=fname(print_table) + " with incomplete rows", color="cyan")
    print()

    # use max_column_width to shorten a cell with long data (str)
    rows = [[1, 2, "3" * 100], [1, "2" * 100, 3]]
    print_table(rows, headers=["a", "b", "c"], max_column_width=10, name=fname(print_table) + " with long cell")
    print()

    # get 3 key statistics from an iterator at once
    with block(fname(stats_of)):
        numbers = [1, 2, 3, 4, 5]
        log_assert(stats_of(numbers)["mean"], 3.0)
        log_assert(stats_of(numbers)["var"], 2.0)

    # curr_time() == str(datetime.now(timezone.utc))[:19]
    with block(fname(curr_time)):
        log(curr_time())
        log(curr_time(breakdown=True))

    async def f(time):
        await asyncio.sleep(time)
        return {"success": True}

    await async_work(f, [{"time": random.random() * 2} for _ in range(8)])
    log("async_work(concurrency=1)")
    await async_work(f, [{"time": 0.3} for _ in range(5)], concurrency=1)


def test_llm():
    with block(fname(build_icl_inputs)):
        sys_msg, resp_format = build_icl_inputs(
            "Let's think about this math problem step by step",
            format={"res": "string"},
            examples=[{"input": "5 + 5", "res": "10"}, {"input": "2 * (3 + 4)", "res": "14"}, {"input": "512 - 112", "res": "400"}],
        )
        prints(resp_format, color="blue")
        log(sys_msg)


async def test():
    await test_core()
    test_llm()


def sync():
    import os
    from src.mbp.info import VERSION

    if os.path.exists("./dist"):
        shutil.rmtree("./dist")

    os.system("python -m build")
    os.system("python -m twine upload --repository pypi dist/* --verbose")

    os.system("git rm --cached -r *")
    os.system("git add .")
    os.system('git commit -a -m "update"')
    os.system("git push origin main")

    for i in range(2):
        os.system("python -m pip install mbp=={}".format(VERSION))


if __name__ == "__main__":
    # run_with_args("test")
    print(sys.argv[1])