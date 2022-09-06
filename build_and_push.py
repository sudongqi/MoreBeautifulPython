import os

os.system("python -m build")
os.system("python -m twine upload --repository pypi dist/* --verbose")
os.system("git commit -a -m \"update\"")
os.system("git push origin main")
os.system("git push wakhi main")
os.system("python -m pip install mbp -U")
