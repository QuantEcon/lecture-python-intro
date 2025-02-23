#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if a directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

TARGET_DIR="$1"

# Ensure the directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

# Find and execute all Python files
echo "Searching for Python files in $TARGET_DIR..."
PYTHON_FILES=$(find "$TARGET_DIR" -type f -name "*.py" | sort)

if [ -z "$PYTHON_FILES" ]; then
    echo "No Python files found in $TARGET_DIR"
    exit 0
fi

# Run each Python file
for file in $PYTHON_FILES; do
    echo " Running: $file"
    MPLBACKEND=Agg python "$file"
done

echo "✅ All scripts ran successfully!"
