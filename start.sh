#!/bin/bash

# Start Dataset Creator Agent Chat UI
# This script starts both the Python agent and the Next.js UI

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Dataset Creator Agent Chat UI...${NC}"

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is required but not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is required but not installed. Please install Node.js 18+ and try again."
    exit 1
fi

# Check AWS credentials
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Warning: AWS credentials not set. Export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to use AWS Bedrock."
    echo "Example:"
    echo "export AWS_ACCESS_KEY_ID=your_access_key"
    echo "export AWS_SECRET_ACCESS_KEY=your_secret_key"
    echo "export AWS_DEFAULT_REGION=your_region"
    echo ""
    echo "Continue without AWS credentials? Agent functionality will be limited. (y/n)"
    read -r answer
    if [ "$answer" != "y" ]; then
        exit 1
    fi
fi

# Check for PostgreSQL environment variables
if [ -z "$POSTGRES_URI" ]; then
    echo "Warning: PostgreSQL connection string not set. Export POSTGRES_URI for persistent conversation state."
    echo "Example:"
    echo "export POSTGRES_URI=postgresql://username:password@localhost:5432/datasets"
    echo ""
    echo "Continue without PostgreSQL? Agent persistence will be disabled. (y/n)"
    read -r answer
    if [ "$answer" != "y" ]; then
        exit 1
    fi
fi

# Install Python dependencies if needed
echo -e "${GREEN}Checking Python dependencies...${NC}"
pip install -q -r requirements.txt > /dev/null

# Install Playwright browsers if needed
echo -e "${GREEN}Installing Playwright browsers...${NC}"
python -c "from playwright.sync_api import sync_playwright; print('Playwright already installed')" || playwright install

# Navigate to ui directory and install dependencies
echo -e "${GREEN}Installing UI dependencies...${NC}"
cd ui || exit
npm install

# Set environment variables for Next.js UI
export NEXT_PUBLIC_AGENT_API_URL=/api/agent
export NEXT_PUBLIC_LANGGRAPH_URL=/api/connect
export NEXT_PUBLIC_AGENT_CONFIG_URL=/api/config
export NEXT_PUBLIC_AGENT_NAME="Dataset Creator Agent"

# Set environment variables for LangGraph and LangSmith
export USE_EXPLICIT_GRAPH=true
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="dataset-creator-agent"
export LANGCHAIN_ENDPOINT=${LANGCHAIN_ENDPOINT:-"https://api.smith.langchain.com"}

# Check if LangSmith API key is available
if [ -z "$LANGCHAIN_API_KEY" ]; then
    echo "Warning: LANGCHAIN_API_KEY not set. LangSmith tracing will be limited."
    echo "For complete tracing, set LANGCHAIN_API_KEY environment variable."
    echo ""
fi

# Start the UI in the background
echo -e "${GREEN}Starting UI server...${NC}"
npm run dev &
UI_PID=$!

# Go back to root directory
cd ..

# Start the Python agent API server
echo -e "${GREEN}Starting dataset agent API server...${NC}"
python dataset_agent.py --api &
AGENT_PID=$!

echo -e "${BLUE}Dataset Creator Agent Chat UI is running!${NC}"
echo -e "Open ${GREEN}http://localhost:3000${NC} in your browser"
echo -e "Dataset Agent API running on ${GREEN}http://localhost:8000${NC}"
echo "Press Ctrl+C to stop all servers"

# Keep the script running and capture Ctrl+C
trap "kill $UI_PID $AGENT_PID; echo -e '\n${BLUE}Shutting down servers...${NC}'; exit" INT
wait