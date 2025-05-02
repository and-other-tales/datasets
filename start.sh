#!/bin/bash

# Start Dataset Creator Agent Chat UI
# This script starts both the LangGraph server and the Next.js UI

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

# Start the LangGraph server in the background
echo -e "${GREEN}Starting LangGraph server...${NC}"
cd ..
npx langgraph dev --config langgraph.json > /tmp/langgraph_output.log 2>&1 &
LANGGRAPH_PID=$!

# Give LangGraph a moment to start
echo -e "${GREEN}Waiting for LangGraph to initialize...${NC}"
sleep 5

# Check if LangGraph started successfully
if ! kill -0 $LANGGRAPH_PID > /dev/null 2>&1; then
    echo -e "${YELLOW}LangGraph failed to start. Check /tmp/langgraph_output.log for details.${NC}"
    exit 1
fi

# Extract Assistant ID from LangGraph logs
ASSISTANT_ID=""
for i in {1..10}; do
    if [ -f "/tmp/langgraph_output.log" ]; then
        ASSISTANT_ID=$(grep -o "Created assistant with ID: \w\+-\w\+-\w\+-\w\+-\w\+" /tmp/langgraph_output.log | grep -o "\w\+-\w\+-\w\+-\w\+-\w\+$" | head -1)
        if [ -n "$ASSISTANT_ID" ]; then
            echo -e "${GREEN}Found Assistant ID: ${ASSISTANT_ID}${NC}"
            break
        fi
    fi
    echo "Waiting for Assistant ID... (attempt $i)"
    sleep 2
done

if [ -z "$ASSISTANT_ID" ]; then
    echo -e "${YELLOW}Warning: Could not find Assistant ID in log file. UI may have limited functionality.${NC}"
fi

# Generate the URL with assistant ID
if [ -n "$ASSISTANT_ID" ]; then
    UI_URL="http://localhost:8080/?apiUrl=http://localhost:2024&assistantId=${ASSISTANT_ID}&chatHistoryOpen=true"
else
    UI_URL="http://localhost:8080"
fi

# Start Nginx proxy
echo -e "${GREEN}Starting Nginx proxy server...${NC}"
nginx -c "$(pwd)/nginx.conf" -g "daemon off;" &
NGINX_PID=$!

# Wait a moment for Nginx to start
sleep 2

# Start the UI
echo -e "${GREEN}Starting UI server...${NC}"
cd ui
npm run dev &
UI_PID=$!

# Display information to the user
echo -e "${BLUE}OtherTales Datasets UI is running!${NC}"
echo -e "Open ${GREEN}${UI_URL}${NC} in your browser"
echo -e "LangGraph server running at ${GREEN}http://localhost:2024${NC}"
echo -e "Assistant ID: ${GREEN}${ASSISTANT_ID}${NC}"
echo "Press Ctrl+C to stop all servers"

# Keep the script running and capture Ctrl+C
trap "kill $UI_PID $LANGGRAPH_PID $NGINX_PID; echo -e '\n${BLUE}Shutting down servers...${NC}'; exit" INT
wait