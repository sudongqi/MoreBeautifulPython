from aetk import *


def main():
    # all log() in this script will print to "./log"
    set_global_logger(file='./log')

    # similar to logging, we can create an independent logger that replace the global logger
    ind_log = Logger(file=sys.stdout)
    ind_log('this is from an independent logger')

    # logger() context manager create a temporarily logger that replace the global logger
    with logger(level=DEBUG, file=sys.stderr, prefix='===> ', log_time=True, log_module=True):
        # log() is the same as print(), this message will be redirected to sys.stderr
        log('this will not be included in "./log"', level=CRITICAL)
        '''
        in sys.stderr:
        2022-08-04 02:06:31  __main__  ===> this will not be included in "./log"
        '''

    # suppress all log by specifying level=SILENT
    with logger(level=SILENT):
        for i in range(1000):
            # no printing happens here
            log(i)

    # use timer() context manager to get execution time
    with timer():
        d = {i: i for i in range(100)}
        for i in range(200000):
            d.get(i, None)
    '''took 0.012994766235351562 seconds'''

    # Workers() is more flexible than multiprocessing.Pool()
    with timer():
        num_workers, num_tasks, fail_rate, exec_time = 4, 8, 0.3, 0.2
        workers = Workers(f=test_f, num_workers=num_workers)
        [workers.add_task((i, fail_rate, exec_time)) for _ in range(num_tasks)]
        [workers.get_res() for _ in range(num_tasks)]
        workers.terminate()
    '''
    started worker-0
    started worker-1
    started worker-2
    started worker-3
    worker-2 failed task-2 : AssertionError('simulated failure (30.0%)')
    worker-2 completed task-4
    worker-1 completed task-1
    worker-3 completed task-3
    worker-0 completed task-0
    worker-1 failed task-7 : AssertionError('simulated failure (30.0%)')
    worker-2 completed task-5
    worker-0 completed task-6
    took 0.5289952754974365 seconds
    '''

    # dir_of(__file__) return the directory of the current file
    this_dir = dir_of(__file__)
    '''C:\\Users\\usr\\AbsolutelyEssentialToolKit'''

    # dir_of() can go up multiple levels
    log(dir_of(lib_path(), level=2))
    '''C:\\Users\\usr\\miniconda3'''

    # path_join() is the same as os.path.join()
    log(path_join(this_dir, 'a', 'b', 'c.file'))
    '''C:\\Users\\usr\\AbsolutelyEssentialToolKit\\a\\b\\c.file'''

    # exec_dir() return the directory you run your python command at
    log(exec_dir())
    '''C:\\Users\\usr\\AbsolutelyEssentialToolKit'''

    # lib_dir() return the path of this python library
    log(lib_path())
    '''C:\\Users\\usr\\miniconda3\\Lib\\site-packages\\aetk.py'''

    log(get_only_file(path_join(dir_of(__file__, 2), 'data')))
    '''C:\\Users\\usr\\data\\file.txt'''

    # get example data from the same directory
    jsonl_file_path = path_join(this_dir, 'data.jsonl')
    json_file_path = path_join(this_dir, 'data.json')

    # print2() is a better pprint.pprint()
    print2(load_json(json_file_path), indent=4)
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

    # print_list() print items of a list in separate lines
    print_list(data)
    '''
    {'id': 1, 'name': 'Jackson', 'age': 43}
    {'id': 2, 'name': 'Zunaira', 'age': 24}
    {'id': 3, 'name': 'Lorelei', 'age': 72}
    '''

    # iterate() can customize iteration procedures
    # for example, sample 10% and report every 3 yield
    vec = [i for i in range(100)]
    for d in iterate(vec, sample_ratio=0.1, progress_interval=3):
        log(d)
    '''
    6
    8
    12
    3/100
    24
    26
    28
    6/100
    40
    85
    94
    9/100
    '''

    # print_table() can adjust column width automatically
    # print_table(rows) == print_list(build_table(rows))
    rows = [d.values() for d in data]
    column_names = list(data[0].keys())
    print_table(rows, column_names=column_names, columns_gap_size=3)
    '''
    id   name      age
    1    Jackson   43
    2    Zunaira   24
    3    Lorelei   72
    '''

    # sep() print a one-line seperator with default character '-' and wing size of 10
    sep('my text', size=10, char='-')
    '''
    ----------my text----------
    '''

    # text_block() context manager enclose the output of an execution
    with text_block(text='table', size=7, char='=', y_gap_size=1):
        print_table(rows, column_names=column_names, columns_gap_size=3)
    '''

    =======table=======
    id   name      age
    1    Jackson   43
    2    Zunaira   24
    3    Lorelei   72
    ===================

    '''

    # get 3 key statistics from an iterator at once
    log(min_max_avg(load_jsonl(jsonl_file_path), key_f=lambda x: x['age']))
    '''(24, 72, 46.333333333333336)'''


if __name__ == '__main__':
    main()
