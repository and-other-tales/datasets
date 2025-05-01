#!/bin/bash

# Start Dataset Creator Agent Chat UI
# This script starts both the Python agent and the Next.js UI

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting OtherTales Datasets UI...${NC}"

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

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo -e "${GREEN}Loading environment variables from .env file...${NC}"
    set -a
    source .env
    set +a
else
    echo -e "${YELLOW}No .env file found. Using environment variables from shell.${NC}"
    echo -e "${YELLOW}You can create a .env file based on env.example.${NC}"
fi

# Check LLM Provider
if [ -z "$LLM_PROVIDER" ]; then
    echo -e "${YELLOW}Warning: LLM_PROVIDER not set. Defaulting to bedrock.${NC}"
    export LLM_PROVIDER="bedrock"
    echo -e "You can set LLM_PROVIDER to: openai, anthropic, bedrock, azure, google, groq, huggingface"
fi

# Verify provider-specific environment variables
case "$LLM_PROVIDER" in
    bedrock)
        if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
            echo "Warning: AWS credentials not set. AWS Bedrock requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
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
        ;;
    openai)
        if [ -z "$OPENAI_API_KEY" ]; then
            echo "Warning: OPENAI_API_KEY not set. OpenAI provider requires an API key."
            echo "Check env.example for required environment variables."
            echo ""
            echo "Continue without OpenAI API key? Agent functionality will be limited. (y/n)"
            read -r answer
            if [ "$answer" != "y" ]; then
                exit 1
            fi
        fi
        ;;
    anthropic)
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo "Warning: ANTHROPIC_API_KEY not set. Anthropic provider requires an API key."
            echo "Check env.example for required environment variables."
            echo ""
            echo "Continue without Anthropic API key? Agent functionality will be limited. (y/n)"
            read -r answer
            if [ "$answer" != "y" ]; then
                exit 1
            fi
        fi
        ;;
    azure)
        if [ -z "$AZURE_OPENAI_API_KEY" ] || [ -z "$AZURE_OPENAI_ENDPOINT" ]; then
            echo "Warning: Azure OpenAI credentials not set. Required environment variables missing."
            echo "Check env.example for required environment variables."
            echo ""
            echo "Continue without Azure OpenAI credentials? Agent functionality will be limited. (y/n)"
            read -r answer
            if [ "$answer" != "y" ]; then
                exit 1
            fi
        fi
        ;;
    google)
        if [ -z "$GOOGLE_API_KEY" ]; then
            echo "Warning: GOOGLE_API_KEY not set. Google provider requires an API key."
            echo "Check env.example for required environment variables."
            echo ""
            echo "Continue without Google API key? Agent functionality will be limited. (y/n)"
            read -r answer
            if [ "$answer" != "y" ]; then
                exit 1
            fi
        fi
        ;;
    groq)
        if [ -z "$GROQ_API_KEY" ]; then
            echo "Warning: GROQ_API_KEY not set. Groq provider requires an API key."
            echo "Check env.example for required environment variables."
            echo ""
            echo "Continue without Groq API key? Agent functionality will be limited. (y/n)"
            read -r answer
            if [ "$answer" != "y" ]; then
                exit 1
            fi
        fi
        ;;
    huggingface)
        if [ -z "$HUGGINGFACE_API_KEY" ] && [ -z "$HUGGINGFACE_MODEL_ID" ]; then
            echo "Warning: HuggingFace credentials not set. Required environment variables missing."
            echo "Check env.example for required environment variables."
            echo ""
            echo "Continue without HuggingFace credentials? Agent functionality will be limited. (y/n)"
            read -r answer
            if [ "$answer" != "y" ]; then
                exit 1
            fi
        fi
        ;;
    *)
        echo "Warning: Unknown LLM_PROVIDER: $LLM_PROVIDER. Check for typos."
        echo "Supported providers: openai, anthropic, bedrock, azure, google, groq, huggingface"
        ;;
esac

# Check for PostgreSQL environment variables
if [ -z "$POSTGRES_URI" ]; then
    echo -e "${YELLOW}Warning: PostgreSQL connection string not set. Export POSTGRES_URI for persistent conversation state.${NC}"
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
export NEXT_PUBLIC_AGENT_NAME="OtherTales Datasets Agent"
# Set absolute URL for the agent endpoint for development (direct access without proxy)
export DATASET_AGENT_URL=http://localhost:2024/agent

# Set environment variables for LangGraph and LangSmith
export USE_EXPLICIT_GRAPH=true
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="othertales-datasets"
export LANGCHAIN_ENDPOINT=${LANGCHAIN_ENDPOINT:-"https://api.smith.langchain.com"}

# Check if LangSmith API key is available
if [ -z "$LANGCHAIN_API_KEY" ]; then
    echo -e "${YELLOW}Warning: LANGCHAIN_API_KEY not set. LangSmith tracing will be limited.${NC}"
    echo "For complete tracing, set LANGCHAIN_API_KEY environment variable."
    echo ""
fi

# Start the UI in the background
echo -e "${GREEN}Starting UI server...${NC}"
npm run dev &
UI_PID=$!

# Go back to root directory
cd ..

# Print currently active LLM provider
echo -e "${GREEN}Using LLM Provider: ${BLUE}$LLM_PROVIDER${NC}"

# Start the Python agent API server
echo -e "${GREEN}Starting OtherTales Datasets API server...${NC}"
# Try to use port 2024 by default, but agent will find a free port if needed
export DATASET_AGENT_PORT=2024
python agent_standalone.py > /tmp/agent_output.log 2>&1 &
AGENT_PID=$!

# Give the agent a moment to start up and find a port
sleep 2

# Read the actual port from the log file
if [ -f /tmp/agent_output.log ]; then
    ACTUAL_PORT=$(grep -o "Starting standalone agent on http://localhost:[0-9]\+" /tmp/agent_output.log | grep -o "[0-9]\+$")
    if [ -n "$ACTUAL_PORT" ]; then
        export DATASET_AGENT_PORT=$ACTUAL_PORT
        export DATASET_AGENT_URL=http://localhost:$ACTUAL_PORT/agent
        echo -e "Dataset agent using port: ${GREEN}$ACTUAL_PORT${NC}"
    fi
fi

echo -e "${BLUE}OtherTales Datasets UI is running!${NC}"
echo -e "Open ${GREEN}http://localhost:3000${NC} in your browser"
echo -e "OtherTales Datasets API running on ${GREEN}http://localhost:${DATASET_AGENT_PORT}${NC}"
echo "Press Ctrl+C to stop all servers"

# Keep the script running and capture Ctrl+C
trap "kill $UI_PID $AGENT_PID; echo -e '\n${BLUE}Shutting down servers...${NC}'; exit" INT
wait