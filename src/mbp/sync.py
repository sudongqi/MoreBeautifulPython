import os
import sys

os.system("git rm --cached -r *")
os.system("git add .")
os.system('git commit -a -m "update"')
os.system(f"git push origin {sys.argv[1] if len(sys.argv) > 1 else 'main'}")
