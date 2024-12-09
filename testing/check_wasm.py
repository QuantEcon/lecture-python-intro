# Testing script to check whether the lectures added in
# wasm_compatible.yml are compatible with WASM.
# This is a soft-check check based on the imports present.
# It is still recommended to test the lecture on the
# https://github.com/QuantEcon/project.lecture-wasm

import os
import yaml
import jupytext

from pyodide.code import find_imports

# This is the list of imports (libraries) that are not supported
# WASM pyodide kernel in Jupyterlite.
UNSUPPORTED_LIBS = {
    'quantecon',
    'numba',
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(CURRENT_DIR)
LECTURES_DIR  = os.path.join(ROOT_DIR, "lectures")
CONFIG_FILE = os.path.join(ROOT_DIR, "wasm_compatible.yml")


def get_wasm_compatible_files():
    """
    Get the list of lectures names from the config file.
    """
    # Load the YAML configuration
    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    return config.get("lectures", [])


def convert_md_to_py_string(md_file_path):
    """
    Convert a .md(myst) file to Python code string without creating a .py file.
    
    Args:
        md_file_path (str): Path to the Markdown file (.md)
        
    Returns:
        str: Python code as a string
    """
    # Read the markdown file as a Jupytext notebook
    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        notebook = jupytext.read(md_file)
    
    # Convert the notebook to Python script format
    py_code = jupytext.writes(notebook, fmt="py")
    
    return py_code


def test_compatibility():
    file_names = get_wasm_compatible_files()
    for file_name in file_names:
        py_string = convert_md_to_py_string(os.path.join(LECTURES_DIR, file_name + ".md"))
        file_imports = find_imports(py_string)
        for file_import in file_imports:
            if file_import in UNSUPPORTED_LIBS:
                error = 'import `{}` in lecture `{}` is not supported by WASM'.format(file_import, file_name)
                raise ValueError(error)

if __name__ == "__main__":
    test_compatibility()
