FROM node:22-alpine AS ui-builder

WORKDIR /ui

# Install dependencies
COPY ui/package*.json ./
RUN npm install --legacy-peer-deps

# Copy all UI files
COPY ui/ .

# Make prebuild.sh executable and run it
RUN chmod +x prebuild.sh
RUN mkdir -p /ui/src/lib/

# Create agent-integration.ts if it doesn't exist (wrapped in shell command to allow failure)
RUN mkdir -p /ui/src/lib/
RUN /bin/sh -c "cp -f ui/src/lib/agent-integration.ts* /ui/src/lib/ 2>/dev/null || echo 'agent-integration.ts not found in source, will be created by prebuild.sh'"

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
ENV NEXT_PUBLIC_AGENT_NAME="OtherTales Datasets Agent"
ENV DATASET_AGENT_URL=http://localhost:8080/agent
ENV USE_EXPLICIT_GRAPH=true

# Set up non-root user for Node
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs && \
    mkdir -p /app/.next /app/public && \
    chown -R nextjs:nodejs /app/.next /app/public

# Install Node.js and other required packages
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directory for GCS mount and set permissions
RUN mkdir -p /gcs && chown nextjs:nodejs /gcs

# Copy Python requirements and install dependencies
COPY requirements.txt .
# Install Python requirements and playwright separately
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install playwright && playwright install --with-deps chromium

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
ENV LANGCHAIN_PROJECT=othertales-datasets
ENV LANGCHAIN_TRACING_V2=true

# Create a new startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Trap SIGTERM and SIGINT to properly shutdown both services\n\
function shutdown() {\n\
  echo "Shutting down services..."\n\
  [ -n "${AGENT_PID}" ] && kill -TERM ${AGENT_PID} || true\n\
  exit 0\n\
}\n\
\n\
trap shutdown SIGTERM SIGINT\n\
\n\
# Start the dataset agent API server\n\
echo "Starting OtherTales Datasets API on port 8080..."\n\
python datasets.py --api &\n\
AGENT_PID=$!\n\
\n\
# Wait for the agent to be ready\n\
echo "Waiting for agent API to be available..."\n\
attempts=0\n\
max_attempts=30\n\
while [ $attempts -lt $max_attempts ]; do\n\
  if curl -s http://localhost:8080/status > /dev/null; then\n\
    echo "Agent API is available!"\n\
    break\n\
  fi\n\
  attempts=$((attempts + 1))\n\
  echo "Waiting for agent API (attempt ${attempts}/${max_attempts})..."\n\
  sleep 1\n\
done\n\
\n\
if [ $attempts -eq $max_attempts ]; then\n\
  echo "WARNING: Agent API did not respond in time, but continuing startup..."\n\
fi\n\
\n\
# Start the UI server\n\
echo "Starting UI server..."\n\
exec node server.js\n' > /app/docker-start.sh && \
chmod +x /app/docker-start.sh

# Start both services
CMD ["/app/docker-start.sh"]