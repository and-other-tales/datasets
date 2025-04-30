#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
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
export DATASET_AGENT_URL=http://localhost:2024/agent

# Set environment variables for LangGraph and LangSmith
export USE_EXPLICIT_GRAPH=true
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="othertales-datasets"
export LANGCHAIN_ENDPOINT=${LANGCHAIN_ENDPOINT:-"https://api.smith.langchain.com"}

# Mount GCS bucket using GCSFuse
echo -e "${GREEN}Mounting GCS bucket using GCSFuse...${NC}"
gcsfuse --debug_gcs --debug_fuse mixture-othertales-co /gcs || {
  echo -e "${RED}Failed to mount GCS bucket. Exiting...${NC}"
  exit 1
}

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

# Start the dataset agent API server
echo -e "${GREEN}Starting Dataset Agent API server on port 2024...${NC}"
python dataset_agent.py --api &
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

# Start Nginx
echo -e "${GREEN}Starting Nginx on port ${PORT}...${NC}"
nginx -g "daemon off;" &
NGINX_PID=$!

echo -e "${BLUE}OtherTales Datasets is running!${NC}"
echo -e "Service available at http://localhost:${PORT}"

# Keep script running until terminated
wait ${NGINX_PID}
