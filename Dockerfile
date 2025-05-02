# Python dependencies builder stage
FROM python:3.12-slim AS python-builder

WORKDIR /app

# Copy Python requirements
COPY requirements.txt .

# Install dependencies into a virtual environment
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    /app/venv/bin/pip install --no-cache-dir playwright && \
    /app/venv/bin/playwright install --with-deps chromium

# Final image
FROM python:3.12-slim

# Set non-interactive frontend and timezone
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PORT=8080 \
    USE_EXPLICIT_GRAPH=true \
    PATH="/app/venv/bin:$PATH"

WORKDIR /app

# Install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python virtual environment from builder
COPY --from=python-builder /app/venv /app/venv

# Copy Python and configuration files
COPY dataset_agent.py llm_utils.py langgraph.json ./ 

# No scripts to make executable

# Expose port
EXPOSE 8080

# Start with langgraph dev
CMD ["langgraph", "dev"]
