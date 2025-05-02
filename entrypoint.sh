#!/bin/bash
set -e

# Print debugging info
echo "PYTHONPATH: $PYTHONPATH"
echo "PWD: $(pwd)"
echo "PORT: $PORT"
echo "HOST: $HOST"

# Start the FastAPI application with uvicorn
exec uvicorn othertales.datasets.dataset_agent:app \
    --host ${HOST:-0.0.0.0} \
    --port ${PORT:-2024} \
    --workers 1 \
    --timeout-keep-alive 75