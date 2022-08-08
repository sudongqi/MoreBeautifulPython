## More Beautiful Python

Make Python more beautiful :) This package includes syntax sugar & tools that you wish were in the standard library.

### Setup

    python -m pip install mbp
    python -m mbp

### Examples
https://github.com/sudongqi/MoreBeautifulPython/blob/main/examples.py

### All functionalities

    __all__ = [

    # Alternative for logging
    'log', 'logger', 'debug', 'error',
    'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'SILENT',
    # Alternative for multiprocessing
    'Workers', 'Worker', 'work', 'test_f',
    # Syntax sugar for pathlib
    'dir_of', 'path_join', 'make_dir', 'this_dir', 'exec_dir', 'lib_path', 'only_file_of',
    # Tools for file loading
    'iterate', 'load_jsonl', 'load_json', 'load_csv', 'load_tsv', 'load_txt', 'save_json', 'save_jsonl', 'open_file',
    # Tools for summarization
    'print2', 'log2', 'enclose', 'enclose_timer', 'print_table', 'build_table', 'print_iter', 'error_msg', 'sep', 'na',
    # Tools for simple statistics
    'timer', 'curr_time', 'avg', 'min_max_avg', 'n_min_max_avg', 'CPU_COUNT',
    # common library
    'sys', 'os', 'random', 'json', 'itertools',

    ]
  






