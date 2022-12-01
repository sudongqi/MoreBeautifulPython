import os
import sys
import subprocess
from src.mbp import log

os.environ['PYTHONPATH'] = 'src'

def run_process(command):
    proc = subprocess.Popen(command,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    out, err = proc.communicate()
    exit_code = proc.returncode
    return exit_code, out, err


def test_log():
    exit_code, out, err = run_process([sys.executable, 'example_log.py'])
    assert exit_code == 0
    assert out.decode() == 'this is from the global logger\n'
    assert err.decode() == ''
