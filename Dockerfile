FROM node:22-alpine AS ui-builder

WORKDIR /ui

# Install dependencies
COPY ui/package*.json ./
RUN npm install --legacy-peer-deps

# Copy UI files
COPY ui/ .

# Create necessary directories for build
RUN mkdir -p /ui/src/lib/ && \
    sh -c "cp -f ui/src/lib/agent-integration.ts* /ui/src/lib/ 2>/dev/null || echo 'agent-integration.ts not found in source'"

# Set environment variables for build
ENV NEXT_PUBLIC_AGENT_API_URL=/api/agent \
    NEXT_PUBLIC_LANGGRAPH_URL=/api/connect \
    NEXT_PUBLIC_AGENT_CONFIG_URL=/api/config

# Build the Next.js app
RUN NODE_OPTIONS=--max-old-space-size=8192 npm run build

# Python dependencies builder stage
FROM python:3.11-slim AS python-builder

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
FROM python:3.11-slim

# Set non-interactive frontend and timezone
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    NODE_ENV=production \
    PORT=8080 \
    NEXT_TELEMETRY_DISABLED=1 \
    NEXT_PUBLIC_AGENT_API_URL=/api/agent \
    NEXT_PUBLIC_LANGGRAPH_URL=/api/connect \
    NEXT_PUBLIC_AGENT_CONFIG_URL=/api/config \
    NEXT_PUBLIC_AGENT_NAME="OtherTales Datasets Agent" \
    DATASET_AGENT_URL=/agent \
    DATASET_AGENT_STATUS_URL=/status \
    DATASET_AGENT_CONFIG_URL=/config \
    USE_EXPLICIT_GRAPH=true \
    PATH="/app/venv/bin:$PATH"

WORKDIR /app

# Install Node.js and Nginx
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    nginx \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/.next /app/public /gcs

# Set up non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs && \
    chown -R nextjs:nodejs /app/.next /app/public /gcs

# Copy Python virtual environment from builder
COPY --from=python-builder /app/venv /app/venv

# Copy built Next.js app
COPY --from=ui-builder --chown=nextjs:nodejs /ui/public ./public
COPY --from=ui-builder --chown=nextjs:nodejs /ui/.next/standalone ./
COPY --from=ui-builder --chown=nextjs:nodejs /ui/.next/static ./.next/static

# Copy Python and configuration files
COPY dataset_agent.py llm_utils.py langgraph.json cloudrun-start.sh nginx.conf env.example ./

# Make scripts executable
RUN chmod +x cloudrun-start.sh

# Expose port
EXPOSE 8080

# Start the integrated service
CMD ["/app/cloudrun-start.sh"]
