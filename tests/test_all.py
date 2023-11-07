import sys
import subprocess
from mbp import shorten_str


def run_process(command):
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    exit_code = proc.returncode
    return exit_code, out, err


def test_log():
    exit_code, out, err = run_process([sys.executable, 'tests/run_log.py'])
    assert exit_code == 0
    NL = '\r\n' if sys.platform == 'win32' else '\n'
    assert out.decode() == f'this is from the global logger{NL}'
    assert err.decode() == ''


def test_shorten():
    assert len(shorten_str("abcdefghi", 6)) == 6
