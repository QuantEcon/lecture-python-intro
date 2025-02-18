import os
import glob
import shutil

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PARENT_DIR)

IN_DIR = os.path.join(ROOT_DIR, 'lectures')
OUT_DIR = os.path.join(PARENT_DIR, 'py_files')

def main():
    shutil.copytree(IN_DIR, OUT_DIR, dirs_exist_ok=True)
    cwd = os.getcwd()
    os.chdir(OUT_DIR)
    cmd = "jupytext --to py *.md"
    os.system(cmd)
    lectures = list(glob.glob(OUT_DIR + '/*.md'))
    for file in lectures:
        os.remove(file)
    os.chdir(cwd)

if __name__ == '__main__':
    main()
