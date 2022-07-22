from aetk import *

# use context manager timer() to get execution time
with timer():
    d = {i: i for i in range(100000)}
    for i in range(200000):
        x = d.get(i, None)

'''
took 0.018950223922729492 seconds
'''

# aetk_dir() inspect the location of the package
file_path = os.path.join(aetk_dir(), 'aetk_test.jsonl')

# load_jsonl()
data = [d for d in load_jsonl(file_path)]

# print_list() print items in list in separate lines
print_list(data)

'''
{'id': 1, 'name': 'Jackson', 'age': 43}
{'id': 2, 'name': 'Zunaira', 'age': 24}
{'id': 3, 'name': 'Lorelei', 'age': 72}
'''

# load_jsonl() can customize loading procedures; sample 50% from the first 2 items
for d in load_jsonl(file_path, take_n=2, sample_ratio=0.5, sample_seed=1234):
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
sep('text')

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
_min, _max, avg = min_max_avg([1, 2, 3, 4, 5])

# the current execution directory
running_dir()

# directory of the file that call this function
dir_of(__file__)
