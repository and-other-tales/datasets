#!/bin/bash
set -e

# Ensure the package is properly installed
pip install -e .

# Print Python path for debugging
echo "PYTHONPATH: $PYTHONPATH"
echo "PWD: $(pwd)"
echo "Listing package directory:"
ls -la /app/src/othertales/datasets

# Start the LangGraph server
exec langgraph dev