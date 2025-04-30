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
FROM ubuntu:24.10

WORKDIR /app

# Set environment variables
ENV NODE_ENV=production
ENV PORT=8080
ENV NEXT_TELEMETRY_DISABLED=1
ENV NEXT_PUBLIC_AGENT_API_URL=/api/agent
ENV NEXT_PUBLIC_LANGGRAPH_URL=/api/connect
ENV NEXT_PUBLIC_AGENT_CONFIG_URL=/api/config
ENV NEXT_PUBLIC_AGENT_NAME="OtherTales Datasets Agent"
ENV DATASET_AGENT_URL=http://localhost:2024/agent
ENV USE_EXPLICIT_GRAPH=true

# Install Node.js, Nginx, and other required packages
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    nginx \
    passwd \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user for Node
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs && \
    mkdir -p /app/.next /app/public && \
    chown -R nextjs:nodejs /app/.next /app/public

# Set the GCSFUSE_REPO environment variable
ENV GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)

# Add the Google Cloud public key to the sources list
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list

# Download the Google Cloud public key
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc

# Update the package list
RUN sudo apt-get update

# Install GCSFuse
RUN sudo apt-get install gcsfuse

# Create directory for GCS mount and set permissions
RUN mkdir -p /gcs && chown nextjs:nodejs /gcs

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install playwright && playwright install --with-deps chromium

# Copy built Next.js app
COPY --from=ui-builder --chown=nextjs:nodejs /ui/public ./public
COPY --from=ui-builder --chown=nextjs:nodejs /ui/.next/standalone ./
COPY --from=ui-builder --chown=nextjs:nodejs /ui/.next/static ./.next/static

# Copy Python and configuration files
COPY dataset_agent.py langgraph.json cloudrun-start.sh nginx.conf ./

# Make scripts executable
RUN chmod +x cloudrun-start.sh

# Expose port
EXPOSE 8080

# Set environment variables from Cloud Run env vars
ENV POSTGRES_URI=$POSTGRES_URI
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV LANGSMITH_TRACING=true
ENV LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
ENV LANGSMITH_API_KEY=$LANGSMITH_API_KEY
ENV LANGSMITH_PROJECT="datasets"

# Start the integrated service (NextJS + LangGraph + Nginx)
CMD ["/app/cloudrun-start.sh"]
