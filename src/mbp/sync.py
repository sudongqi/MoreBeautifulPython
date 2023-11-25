import os

os.system("git rm --cached -r *")
os.system("git add .")
os.system("git commit -a -m \"update\"")
os.system("git push origin main")
