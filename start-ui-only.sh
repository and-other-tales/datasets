#!/bin/bash

# Start only the UI component, connecting to agent via API
# This script assumes the agent is already running or will be started separately

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting OtherTales Datasets UI (UI-only mode)...${NC}"

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo -e "${GREEN}Loading environment variables from .env file...${NC}"
    set -a
    source .env
    set +a
else
    echo -e "${YELLOW}No .env file found. Using environment variables from shell.${NC}"
fi

# Navigate to ui directory and install dependencies
echo -e "${GREEN}Installing UI dependencies...${NC}"
cd ui || exit
npm install

# Set environment variables for Next.js UI
export NEXT_PUBLIC_AGENT_API_URL=/api/agent
export NEXT_PUBLIC_AGENT_CONFIG_URL=/api/config
export NEXT_PUBLIC_AGENT_NAME="OtherTales Datasets Agent"

# Start the UI server
echo -e "${GREEN}Starting UI server...${NC}"
echo -e "${YELLOW}NOTE: This script only starts the UI - not the agent.${NC}"
echo -e "${YELLOW}To start the agent separately, run: python agent_standalone.py${NC}"
echo -e "Open ${GREEN}http://localhost:3000${NC} in your browser"

# Start the UI
npm run dev