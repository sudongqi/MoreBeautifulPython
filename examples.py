from aetk import *


def main():
    # all log() in this script will print to file at path "./log"
    set_global_logger(file='./log')

    # similar to logging, we can create an independent logger that replace the global logger
    my_log = Logger(file=sys.stdout)
    # log() include all functionality of print()
    my_log('this is from an independent logger', end='\n')

    # logger() context manager create a temporary logger that replace the global logger
    with logger(level=DEBUG, file=sys.stderr, prefix='===> ', log_time=True, log_module=True):
        # this message will be redirected to sys.stderr
        log('this is from a temporary logger', level=CRITICAL)
    '''
    2022-08-05 01:27:34  __main__  ===> this is from a temporary logger
    '''

    # suppress all logs by specifying level=SILENT
    with logger(level=SILENT):
        for i in range(1000):
            # all loggings are suppressed
            log(i)

    # use timer() context manager to get execution time
    with timer():
        d = {i: i for i in range(100)}
        for i in range(200000):
            d.get(i, None)
    '''took 0.012994766235351562 seconds'''

    # enclose() generate two text separators that enclose the execution
    with enclose(text='box', size_x=10, size_y=1, char='=', timer=False):
        log('content')
    '''
    ==========box==========
    content
    =======================
    '''

    # timer_enclose() == enclose(timer=True)
    with timer_enclose():
        # iterate() can customize iteration procedures
        # for example, sample 10% and report every 3 yield from the first 100 samples
        for d in iterate(range(1000), take_n=100, sample_ratio=0.1, progress_interval=3):
            log(d)
    '''
    ====================
    4
    13
    22
    3/1000
    36
    59
    64
    6/1000
    65
    77
    ====================
    took 0.0019941329956054688 seconds
    '''

    # Workers() is more flexible than multiprocessing.Pool()
    n_task = 8
    with timer_enclose('Workers()'):
        workers = Workers(f=test_f, num_workers=4, progress=True)
        [workers.add_task({'x': i, 'fail_rate': 0.3}) for _ in range(n_task)]
        [workers.get_res() for _ in range(n_task)]
        workers.terminate()
    '''
    ==========Workers()==========
    started worker-0
    started worker-1
    started worker-2
    started worker-3
    worker-1 completed task-1
    worker-0 completed task-0
    worker-3 completed task-3
    worker-2 failed task-2 : AssertionError('simulated failure (30.0%)')
    worker-3 completed task-7
    worker-2 completed task-5
    worker-0 failed task-6 : AssertionError('simulated failure (30.0%)')
    worker-1 completed task-4
    terminated 4 workers
    =============================
    took 0.5016000270843506 seconds
    '''

    # similarly, we can use work() to process tasks from an iterator
    with timer_enclose('work()'):
        for r in work(f=test_f, tasks=iter([{'x': i, 'fail_rate': 0.5} for i in range(n_task)]), ordered=True):
            log(r)
    '''
    ==========work()==========
    {'worker_id': 2, 'task_id': 0, 'res': 0}
    {'worker_id': 0, 'task_id': 1, 'res': 2}
    {'worker_id': 1, 'task_id': 2, 'res': None, 'error': "AssertionError('simulated failure (50.0%)')"}
    {'worker_id': 3, 'task_id': 3, 'res': 6}
    {'worker_id': 4, 'task_id': 4, 'res': None, 'error': "AssertionError('simulated failure (50.0%)')"}
    {'worker_id': 5, 'task_id': 5, 'res': None, 'error': "AssertionError('simulated failure (50.0%)')"}
    {'worker_id': 6, 'task_id': 6, 'res': None, 'error': "AssertionError('simulated failure (50.0%)')"}
    {'worker_id': 7, 'task_id': 7, 'res': 14}
    ==========================
    took 0.3321056365966797 seconds
    '''

    with enclose('path'):
        # this_dir() return the directory of the current file
        log(this_dir())
        # dir_of() can move up multiple directories
        log(dir_of(__file__, level=2))
        # path_join() == os.path.join()
        log(path_join(this_dir(), 'a', 'b', 'c.file'))
        # exec_dir() return the directory where you run your python command
        log(exec_dir())
        # lib_dir() return the path of the aetk library
        log(lib_path())
        # get_only_file() return the path of the only file in a folder
        log(get_only_file(path_join(this_dir(2), 'data')))
    '''
    ==========path==========
    C:\\Users\sudon\AbsolutelyEssentialToolKit
    C:\\Users\sudon
    C:\\Users\sudon\AbsolutelyEssentialToolKit\a\b\c.file
    C:\\Users\sudon\AbsolutelyEssentialToolKit
    C:\\Users\sudon\AbsolutelyEssentialToolKit\src\aetk.py
    C:\\Users\sudon\data\file.txt
    ========================
    '''

    # get data.jsonl & data.json from the current directory
    jsonl_file_path = path_join(this_dir(), 'data.jsonl')
    json_file_path = path_join(this_dir(), 'data.json')

    # print2() is a superior pprint.pprint()
    print2(load_json(json_file_path), indent=4)
    # log2() is the logging version of the print2()
    log2(load_json(json_file_path), indent=4)
    '''
    {
        "quiz": {
            "maths": {
                "q1": {
                    "question": "5 + 7 = ?",
                    "options": [
                        "10",
                        "11",
                        "12",
                        "13"
                    ],
                    "answer": "12"
                }
            }
        }
    }
    '''

    # load_jsonl() return an iterator of dictionary
    data = list(load_jsonl(jsonl_file_path))

    # print_iter() print items from an iterator one by one
    with enclose('print_list()'):
        print_iter(data)
    '''
    ==========print_iter()==========
    {'id': 1, 'name': 'Jackson', 'age': 43}
    {'id': 2, 'name': 'Zunaira', 'age': 24}
    {'id': 3, 'name': 'Lorelei', 'age': 72}
    ================================
    '''

    # print_table() can adjust column width automatically
    # print_table(rows) == print_iter(build_table(rows))
    with enclose('print_table()'):
        rows = [d.values() for d in data]
        column_names = list(data[0].keys())
        print_table(rows, column_names=column_names, space=3)
    '''
    ==========print_table()==========
    id   name      age
    1    Jackson   43
    2    Zunaira   24
    3    Lorelei   72
    =================================
    '''

    # get 3 key statistics from an iterator at once
    log(min_max_avg(load_jsonl(jsonl_file_path), key_f=lambda x: x['age']))
    '''(24, 72, 46.333333333333336)'''


if __name__ == '__main__':
    main()
