from aetk import *


def run_examples():
    # use context manager timer() to get execution time
    with timer():
        d = {i: i for i in range(100000)}
        for i in range(200000):
            x = d.get(i, None)
    '''
    took 0.018950223922729492 seconds
    '''

    # dir_of(__file__) return the directory of the current file
    this_dir = dir_of(__file__)

    # path_join == os.path.join
    jsonl_file_path = path_join(this_dir, 'example_data.jsonl')
    json_file_path = path_join(dir_of(__file__), 'example_data.json')

    # print2 is better version of pprint
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

    # print_list() print items in list in separate lines
    print_list(data)
    '''
    {'id': 1, 'name': 'Jackson', 'age': 43}
    {'id': 2, 'name': 'Zunaira', 'age': 24}
    {'id': 3, 'name': 'Lorelei', 'age': 72}
    '''

    # load_jsonl() can customize loading procedures;
    # for example, sample 50% from the first 2 items using seed 1234 with no compression
    for d in load_jsonl(jsonl_file_path, take_n=2, sample_ratio=0.5, sample_seed=1234, progress=True, compression=None):
        print(d)
    '''
    {'id': 2, 'name': 'Zunaira', 'age': 24}
    '''

    # print table with dynamic column width
    rows = [d.values() for d in data]
    column_names = list(data[0].keys())
    print_table(rows, column_names, columns_gap_size=3)
    '''
    id   name      age
    1    Jackson   43
    2    Zunaira   24
    3    Lorelei   72
    '''

    # one-liner seperator with default character '-' and wing size of 10
    sep('text', size=10, char='-')
    '''
    ----------text----------
    '''

    # text block that enclose the execution
    with text_block(text='table', size=7, char='=', y_gap_size=1):
        print_table(rows, column_names, columns_gap_size=3)
    '''

    =======table=======
    id   name      age
    1    Jackson   43
    2    Zunaira   24
    3    Lorelei   72
    ===================

    '''

    # get statistics of a int/float list in one for-loop
    print(min_max_avg([1, 2, 3, 4, 5]))
    '''
    (1, 5, 3.0)
    '''

    # lib_dir() return the directory of your python libraries
    print(lib_dir())

    # exec_dir() return the directory you run your python command at
    print(exec_dir())

    # workers class is more flexible than multiprocessing.Pool()
    with timer():
        num_workers, num_tasks, fail_rate, exec_time = 4, 8, 0.3, 0.5
        workers = Workers(f=test_f, num_workers=num_workers)
        [workers.add_task((i, fail_rate, exec_time)) for _ in range(num_tasks)]
        [workers.get_res() for _ in range(num_tasks)]
        workers.terminate()
    '''
    worker-0 started
    worker-1 started
    worker-2 started
    worker-3 started
    worker-0 failed: AssertionError('simulated failure')
    worker-2 completed
    worker-0 completed
    worker-1 completed
    worker-3 completed
    worker-3 failed: AssertionError('simulated failure')
    worker-2 completed
    worker-0 completed
    took 1.1288304328918457 seconds
    '''


if __name__ == '__main__':
    run_examples()
