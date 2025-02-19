import os
import glob
import ast
import importlib.util
import micropip
import asyncio
import sys

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PARENT_DIR)

PY_FILES_DIR = os.path.join(PARENT_DIR, 'py_files')
# TODO: Make this list empty
SKIP_FILES = [
    'short_path.py',
    'inflation_history.py',
    'lp_intro.py', # remove ortools
]

def get_imported_libraries(file_path):
    """Extracts all imported libraries from a Python file."""
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    imports = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])  # Get the top-level package
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])

    return sorted(imports)


def is_library_installed(library_name):
    """Checks if a given library is installed."""
    return importlib.util.find_spec(library_name) is not None


async def install_missing_libraries(file_path, previously_installed):
    """Finds missing libraries and installs them using micropip."""
    for skip_file in SKIP_FILES:
        if file_path.endswith(skip_file):
            return
    print(f"Installing missing libraries for file: {file_path}")
    imported_libraries = get_imported_libraries(file_path)
    
    missing_libraries = []
    for lib in imported_libraries:
        if lib not in previously_installed and not is_library_installed(lib):
            missing_libraries.append(lib)
    
    if len(missing_libraries) == 0:
        print(f"All required libraries are already installed in {file_path}")
        return

    for lib in missing_libraries:
        print(f"Installing missing libraries: {lib}")
        await micropip.install(lib)
        previously_installed.add(lib)


async def main():
    lectures_py = list(glob.glob(PY_FILES_DIR + '/*.py'))
    previously_installed = set()
    for file in lectures_py:
        try:
            await install_missing_libraries(file, previously_installed)
        except Exception as e:
            print(f"failed to install library from file: {file}")
            sys.exit(1)


if __name__ == '__main__':
    if asyncio.get_event_loop().is_running():
        # If an event loop is already running, use create_task()
        task = asyncio.create_task(main())  
        asyncio.get_event_loop().run_until_complete(task)
    else:
        # If no event loop is running, use asyncio.run()
        asyncio.run(main())
