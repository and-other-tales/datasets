#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting OtherTales Datasets with Nginx Reverse Proxy...${NC}"

# Trap SIGTERM and SIGINT to properly shutdown all services
function shutdown() {
  echo "Shutting down services..."
  kill -TERM ${UI_PID} ${LANGGRAPH_PID} ${NGINX_PID} 2>/dev/null || true
  exit 0
}

trap shutdown SIGTERM SIGINT

# Check environment variables
if [ -z "$PORT" ]; then
  echo "PORT environment variable is not set, defaulting to 8080"
  export PORT=8080
fi

# Configure environment variables for Next.js UI
export NEXT_PUBLIC_AGENT_API_URL=/api/agent
export NEXT_PUBLIC_LANGGRAPH_URL=/api/connect
export NEXT_PUBLIC_AGENT_CONFIG_URL=/api/config
export NEXT_PUBLIC_AGENT_NAME="OtherTales Datasets Agent"
# Use relative paths for Cloud Run compatibility
export DATASET_AGENT_URL=/agent
export DATASET_AGENT_STATUS_URL=/status
export DATASET_AGENT_CONFIG_URL=/config

# Set environment variables for LangGraph and LangSmith
export USE_EXPLICIT_GRAPH=true
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="othertales-datasets"
export LANGCHAIN_ENDPOINT=${LANGCHAIN_ENDPOINT:-"https://api.smith.langchain.com"}

# GCS bucket is automatically mounted by Cloud Run
echo -e "${GREEN}Using Cloud Run GCS bucket mount at /gcs...${NC}"
# Ensure the directory is accessible
if [ ! -d "/gcs" ]; then
  echo -e "${RED}GCS mount directory not found. Creating...${NC}"
  mkdir -p /gcs
fi

# Check if GCS mount is working by testing if the directory is accessible
if [ ! -w "/gcs" ]; then
  echo -e "${RED}Warning: GCS mount at /gcs is not writable.${NC}"
else
  # Try to list the contents to confirm the mount is working
  if ! ls -la /gcs > /dev/null 2>&1; then
    echo -e "${RED}Warning: Unable to list contents of /gcs. Mount may not be working correctly.${NC}"
    echo -e "${GREEN}Continuing with startup anyway...${NC}"
  else
    echo -e "${GREEN}GCS mount at /gcs is working correctly.${NC}"
  fi
fi

# Start the dataset agent API server first
echo -e "${GREEN}Starting Dataset Agent API server on port 2024...${NC}"
/app/venv/bin/python dataset_agent.py --api > /tmp/langgraph_output.log 2>&1 &
LANGGRAPH_PID=$!

# Wait for the dataset agent API server to be ready
echo "Waiting for Dataset Agent API server to be available..."
attempts=0
max_attempts=30
while [ $attempts -lt $max_attempts ]; do
  if curl -s http://localhost:2024/status > /dev/null; then
    echo "Dataset Agent API server is available!"
    break
  fi
  attempts=$((attempts + 1))
  echo "Waiting for Dataset Agent API server (attempt ${attempts}/${max_attempts})..."
  sleep 1
done

if [ $attempts -eq $max_attempts ]; then
  echo -e "${RED}Dataset Agent API server failed to start within the expected time.${NC}"
  cat /tmp/langgraph_output.log
  exit 1
fi

# Extract Assistant ID from LangGraph logs
ASSISTANT_ID=""
for i in {1..10}; do
  if [ -f "/tmp/langgraph_output.log" ]; then
    ASSISTANT_ID=$(grep -o "Created assistant with ID: \w\+-\w\+-\w\+-\w\+-\w\+" /tmp/langgraph_output.log | grep -o "\w\+-\w\+-\w\+-\w\+-\w\+$" | head -1)
    if [ -n "$ASSISTANT_ID" ]; then
      echo -e "${GREEN}Found Assistant ID: ${ASSISTANT_ID}${NC}"
      
      # Store the Assistant ID for the UI to use
      export NEXT_PUBLIC_ASSISTANT_ID=$ASSISTANT_ID
      
      # Create a small JavaScript file to inject the Assistant ID into the UI
      mkdir -p /app/public
      cat > /app/public/assistant-config.js << EOF
window.ASSISTANT_ID = "$ASSISTANT_ID";
window.API_URL = "/";
EOF
      
      break
    fi
  fi
  echo "Waiting for Assistant ID... (attempt $i)"
  sleep 2
done

if [ -z "$ASSISTANT_ID" ]; then
  echo -e "${YELLOW}Warning: Could not find Assistant ID in log file. UI may have limited functionality.${NC}"
fi

# Start the Next.js UI
echo -e "${GREEN}Starting UI server on port 3000...${NC}"
cd /app
node server.js &
UI_PID=$!

# Wait for the UI to be ready
echo "Waiting for UI to be available..."
attempts=0
max_attempts=30
while [ $attempts -lt $max_attempts ]; do
  if curl -s http://localhost:3000 > /dev/null; then
    echo "UI is available!"
    break
  fi
  attempts=$((attempts + 1))
  echo "Waiting for UI to start (attempt ${attempts}/${max_attempts})..."
  sleep 1
done

# Configure and start Nginx
echo -e "${GREEN}Updating Nginx configuration for port ${PORT}...${NC}"
# Create Nginx config from template if needed
cat > /etc/nginx/nginx.conf << EOF
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;

events {
    worker_connections 1024;
}

http {
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    server_tokens off;

    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    access_log /dev/stdout;
    error_log /dev/stderr;

    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_buffers 16 8k;
    gzip_http_version 1.1;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    server {
        listen ${PORT};
        server_name localhost;
        
        # Default location - route to Next.js UI on port 3000
        location / {
            proxy_pass http://127.0.0.1:3000;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host \$host;
            proxy_cache_bypass \$http_upgrade;
            # Increase timeouts for long-running requests
            proxy_read_timeout 300s;
        }
        
        # Explicitly handle Next.js static files with proper caching
        location /_next/static/ {
            proxy_pass http://127.0.0.1:3000/_next/static/;
            proxy_http_version 1.1;
            proxy_set_header Host \$host;
            proxy_cache_bypass \$http_upgrade;
            # Add caching for static assets
            expires 30d;
            add_header Cache-Control "public, max-age=2592000";
            access_log off;
        }
    
        # Special case for static assets in public folder
        location /public/ {
            proxy_pass http://127.0.0.1:3000/public/;
            proxy_http_version 1.1;
            proxy_set_header Host \$host;
            proxy_cache_bypass \$http_upgrade;
            # Add caching for public assets
            expires 30d;
            add_header Cache-Control "public, max-age=2592000";
            access_log off;
        }
        
        # Route LangGraph API endpoints to LangGraph server on port 2024
        # /assistants, /threads, /runs, and /store for OpenAI-compatible API
        location ~ ^/(assistants|threads|runs|store)/ {
            proxy_pass http://127.0.0.1:2024;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host \$host;
            proxy_cache_bypass \$http_upgrade;
            # Increase timeouts for LangGraph operations
            proxy_read_timeout 300s;
        }
        
        # Pass Next.js API routes to the Next.js server
        location /api/ {
            proxy_pass http://127.0.0.1:3000;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host \$host;
            proxy_cache_bypass \$http_upgrade;
            # Increase timeouts for API operations
            proxy_read_timeout 300s;
        }
        
        # Special case for dataset agent status endpoint
        location /status {
            proxy_pass http://127.0.0.1:2024;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host \$host;
            proxy_cache_bypass \$http_upgrade;
        }
        
        # Special case for dataset agent endpoint
        location /agent {
            proxy_pass http://127.0.0.1:2024;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host \$host;
            proxy_cache_bypass \$http_upgrade;
            # Increase timeouts for agent operations
            proxy_read_timeout 300s;
        }
    }
}
EOF

# Start Nginx
echo -e "${GREEN}Starting Nginx on port ${PORT}...${NC}"
nginx -g "daemon off;" &
NGINX_PID=$!

# Generate the complete URL
if [ -n "$ASSISTANT_ID" ]; then
  UI_URL="http://localhost:${PORT}/?apiUrl=/&assistantId=${ASSISTANT_ID}&chatHistoryOpen=true"
else
  UI_URL="http://localhost:${PORT}"
fi

echo -e "${BLUE}OtherTales Datasets is running!${NC}"
echo -e "Service available at ${GREEN}${UI_URL}${NC}"
echo -e "Assistant ID: ${GREEN}${ASSISTANT_ID:-"Not available"}${NC}"

# Keep script running until terminated
wait ${NGINX_PID}