FROM python:3.12-slim AS base

WORKDIR /app

# Set environment variables
ENV PORT=2024
ENV HOST=0.0.0.0
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
COPY env.example .env

# Install Python dependencies and local package
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir playwright && playwright install --with-deps chromium

# Ensure Python can find the package
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose port
EXPOSE ${PORT}

# Add healthcheck for Docker
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

# Add labels for Google Cloud Run startup probe
LABEL com.google.cloud.run.startup-probe.path="/startup" \
      com.google.cloud.run.startup-probe.period.seconds="5" \
      com.google.cloud.run.startup-probe.timeout.seconds="3" \
      com.google.cloud.run.startup-probe.failure-threshold="5"

# Copy and make entrypoint script executable
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Start using the entrypoint script
CMD ["/app/entrypoint.sh"]