from .core import *

VERSION = '1.6.3'

with enclose('More Beautiful Python'):
    rows = [
        ['examples', 'https://github.com/sudongqi/MoreBeautifulPython/blob/main/examples.py'],
        ['execution_directory', run_dir()],
        ['library_path', lib_path()],
        ['cpu_count', CPU_COUNT],
        ['version', VERSION]
    ]
    print_table(rows)
