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
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and set up the package structure
RUN mkdir -p /app/src/othertales/datasets

# Copy package files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY langgraph.json ./

# Install Python dependencies and local package
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir playwright && playwright install --with-deps chromium

# Expose port
EXPOSE 2024

# Start LangGraph in dev mode
CMD ["langgraph", "dev"]