import os
import shutil
from mbp.info import VERSION

if os.path.exists('./dist'):
    shutil.rmtree('./dist')

os.system("python -m build")
os.system("python -m twine upload --repository pypi dist/* --verbose")
os.system("git commit -a -m \"update\"")
os.system("git push origin main")

for i in range(2):
    os.system("python -m pip install mbp=={}".format(VERSION))
