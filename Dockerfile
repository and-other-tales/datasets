FROM python:3.12-slim AS base

WORKDIR /app

# Set environment variables
ENV PORT=2024
ENV PYTHONUNBUFFERED=1
ENV NEXT_TELEMETRY_DISABLED=1
ENV LANGSMITH_TRACING=true
ENV LANGSMITH_ENDPOINT="https://api.smith.langchain.com"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY requirements.txt pyproject.toml ./
COPY dataset_agent.py langgraph.json llm_utils.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir playwright && playwright install --with-deps chromium

# Install the local package
RUN pip install -e .

# Expose port
EXPOSE 2024

# Start LangGraph in dev mode
#CMD ["python", "dataset_agent.py", "--api"]
CMD ["langgraph", "dev"]