## AbsolutelyEssentialToolKit

Make Python more beautiful :) This package includes syntax sugar & tools that you wish were in the standard library.

### Setup

    pip install aetk

### Quick Test

    >>> from aetk import *
    >>> print_iter(work(f=test_f, tasks=iter({'x': i} for i in range(32))))

### [examples.py](https://github.com/sudongqi/AbsolutelyEssentialToolKit/blob/main/examples.py)

* multi processes
    * work, Worker, Workers
* logging
    * log, log2, logger, Logger, set_global_logger
* paths
    * dir_of, path_join, make_dir, this_dir, exec_dir, lib_dir
* files
    * iterate, load_jsonl, load_json, load_csv, load_tsv, load_txt, save_json, save_jsonl, open_file
* summarization
    * print2, print_table, print_iter, error_msg, build_table, enclose, timer_enclose, sep, na
* statistics
    * timer, curr_time, min_max_avg, n_min_max_avg 
  






