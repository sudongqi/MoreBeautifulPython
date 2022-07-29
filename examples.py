from aetk import timer, Workers, test_f, min_max_avg, \
    print2, print_list, print_table, sep, text_block, \
    load_jsonl, load_json, dir_of, lib_path, exec_dir, path_join, get_only_file


def main():
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
    print(dir_of(lib_path(), level=2))
    '''C:\\Users\\usr\\miniconda3'''

    # path_join() is the same as os.path.join()
    print(path_join(this_dir, 'a', 'b', 'c.file'))
    '''C:\\Users\\usr\\AbsolutelyEssentialToolKit\\a\\b\\c.file'''

    # exec_dir() return the directory you run your python command at
    print(exec_dir())
    '''C:\\Users\\usr\\AbsolutelyEssentialToolKit'''

    # lib_dir() return the path of this python library
    print(lib_path())
    '''C:\\Users\\usr\\miniconda3\\Lib\\site-packages\\aetk.py'''

    print(get_only_file(path_join(dir_of(__file__, 2), 'data')))
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

    # load_jsonl() can customize loading procedures
    # for example, sample 50% using seed 1234 from the first 2 items
    for d in load_jsonl(jsonl_file_path, take_n=2, sample_ratio=0.5, sample_seed=1234, progress=True, compression=None):
        print(d)
    '''
    {'id': 2, 'name': 'Zunaira', 'age': 24}
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
    print(min_max_avg(load_jsonl(jsonl_file_path), key_f=lambda x: x['age']))
    '''(24, 72, 46.333333333333336)'''


if __name__ == '__main__':
    main()
