FROM node:22-alpine AS ui-builder

WORKDIR /ui

# Install dependencies
COPY ui/package*.json ./
RUN npm install --legacy-peer-deps

# Copy all UI files
COPY ui/ .

# Make prebuild.sh executable and run it
RUN chmod +x prebuild.sh || echo "No prebuild.sh script found, creating directory manually"
RUN mkdir -p /ui/src/lib/ && ls -la /ui/src/lib/

# Run prebuild script for environment setup
RUN sh -c "./prebuild.sh"

# Log the contents of the lib directory
RUN echo "Contents of /ui/src/lib after prebuild:" && ls -la /ui/src/lib/

# Set environment variables for build
ENV NEXT_PUBLIC_AGENT_API_URL=/api/agent
ENV NEXT_PUBLIC_LANGGRAPH_URL=/api/connect
ENV NEXT_PUBLIC_AGENT_CONFIG_URL=/api/config

# Build the Next.js app with increased memory
RUN NODE_OPTIONS=--max-old-space-size=4096 npm run build

# Final image
FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV NODE_ENV=production
ENV PORT=8080
ENV NEXT_TELEMETRY_DISABLED=1
ENV NEXT_PUBLIC_AGENT_API_URL=/api/agent
ENV NEXT_PUBLIC_LANGGRAPH_URL=/api/connect
ENV NEXT_PUBLIC_AGENT_CONFIG_URL=/api/config
ENV NEXT_PUBLIC_AGENT_NAME="Dataset Creator Agent"
ENV DATASET_AGENT_URL=http://localhost:8080/agent
ENV USE_EXPLICIT_GRAPH=true

# Set up non-root user for Node
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs && \
    mkdir -p /app/.next /app/public && \
    chown -R nextjs:nodejs /app/.next /app/public

# Install Node.js
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directory for GCS mount and set permissions
RUN mkdir -p /gcs && chown nextjs:nodejs /gcs

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "from playwright.sync_api import sync_playwright" || playwright install --with-deps chromium

# Copy built Next.js app
COPY --from=ui-builder --chown=nextjs:nodejs /ui/public ./public
COPY --from=ui-builder --chown=nextjs:nodejs /ui/.next/standalone ./
COPY --from=ui-builder --chown=nextjs:nodejs /ui/.next/static ./.next/static

# Copy Python files
COPY datasets.py start.sh ./

# Make start.sh executable
RUN chmod +x start.sh

# Expose port
EXPOSE 8080

# Set environment variables from Cloud Run env vars
ENV POSTGRES_URI=$POSTGRES_URI
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION
ENV LANGCHAIN_API_KEY=$LANGCHAIN_API_KEY
ENV LANGCHAIN_ENDPOINT=$LANGCHAIN_ENDPOINT
ENV LANGCHAIN_PROJECT=dataset-creator-agent
ENV LANGCHAIN_TRACING_V2=true

# Create a new startup script
RUN echo '#!/bin/bash\n\
# Start the dataset agent API server\n\
python datasets.py --api &\n\
AGENT_PID=$!\n\
# Start the UI server\n\
node server.js\n' > /app/docker-start.sh && \
chmod +x /app/docker-start.sh

# Start both services
CMD ["/app/docker-start.sh"]