#!/bin/bash

# Check if task name is provided
if [ -z "$1" ]; then
    echo "Error: Task name is required"
    echo "Usage: ./task_generator.sh <task_name>"
    exit 1
fi

TASK_NAME=$1
TASK_DIR="$TASK_NAME"

# Create task directory
echo "Creating directory: $TASK_DIR"
mkdir -p "$TASK_DIR"

# Create empty notebook files with proper JSON structure
echo "Creating notebook files..."

# Create final_notebook.ipynb
cat > "$TASK_DIR/final_notebook.ipynb" << 'EOF'
{
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create initial_notebook.ipynb
cat > "$TASK_DIR/initial_notebook.ipynb" << 'EOF'
{
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create empty test_notebook.py
touch "$TASK_DIR/test_notebook.py"

echo "Files created successfully"

# Change to task directory
cd "$TASK_DIR" || exit 1

# Create virtual environment with uv
echo "Creating virtual environment with Python 3.10.13..."
uv venv .venv --python 3.10.13

# Activate virtual environment and install dependencies
echo "Installing dependencies..."
uv pip install nbformat pytest nbconvert numpy pandas scipy sympy scikit-learn torch tensorflow ipykernel matplotlib

echo ""
echo "✓ Task setup complete!"
echo "Directory: $TASK_DIR"
echo "Files created:"
echo "  - final_notebook.ipynb"
echo "  - initial_notebook.ipynb"
echo "  - test_notebook.py"
echo ""
echo "To activate the virtual environment, run:"
echo "  cd $TASK_DIR && source .venv/bin/activate"