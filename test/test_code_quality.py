import subprocess
import os
import pathlib
import shutil


ROOT_DIR = pathlib.Path(os.path.abspath(__file__)).parents[1]
LECTURE_DIR = os.path.join(ROOT_DIR, 'lectures')


def check_unused_imports(py_files_dir):
    # Use flake8 to check for unused imports (F401)
    ret = subprocess.run(['flake8', '--select', 'F401', py_files_dir],
                         stdout=subprocess.PIPE)
    if ret.returncode != 0:
        print(ret.stdout.decode('utf-8'))
        raise Exception("Please remove the unused imports")
    print("Passed: Code Quality tests")


if __name__ == "__main__":
    test_dir = os.path.join(ROOT_DIR, '__test_code_quality_tmp')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for file_name in os.listdir(LECTURE_DIR):
        if file_name.endswith(".md"):
            base_file_name = pathlib.Path(file_name).stem
            cmd = ["jupytext", "--to", "py",
                f"lectures/{file_name}", "-o",
                f"{test_dir}/{base_file_name}.py"]
            ret = subprocess.run(cmd, stdout=subprocess.PIPE)
            if ret.returncode != 0:
                shutil.rmtree(test_dir)
                print(ret.stdout.decode('utf-8'))
                raise Exception(f"jupytext failed for file {file_name}")

    try:
        check_unused_imports(test_dir)
    finally:
        shutil.rmtree(test_dir)
