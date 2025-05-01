#!/bin/bash

# Start Dataset Creator Agent Chat UI with Nginx proxy
# This script starts the Python agent, the Next.js UI, and configures nginx

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -i:$port -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill process using a specific port
kill_port_process() {
    local port=$1
    local pid=$(lsof -i:$port -t)
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Port $port is in use by PID $pid. Attempting to terminate...${NC}"
        kill -15 $pid
        sleep 1
        if check_port $port; then
            echo -e "${RED}Failed to terminate process on port $port with SIGTERM. Using SIGKILL...${NC}"
            kill -9 $pid
            sleep 1
            if check_port $port; then
                echo -e "${RED}Failed to free up port $port. Please terminate the process manually.${NC}"
                return 1
            fi
        fi
        echo -e "${GREEN}Successfully freed port $port${NC}"
    fi
    return 0
}

echo -e "${BLUE}Starting OtherTales Datasets UI with Nginx proxy...${NC}"

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

# Check if nginx is installed
if ! command -v nginx &> /dev/null; then
    echo -e "${RED}Nginx is required but not installed.${NC}"
    echo -e "Please install nginx with: ${GREEN}sudo apt-get install -y nginx${NC}"
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

# Check if port 8080 is in use and free it if necessary
if check_port 8080; then
    echo -e "${YELLOW}Port 8080 is already in use. Attempting to free it...${NC}"
    if ! kill_port_process 8080; then
        echo -e "${RED}Failed to free port 8080. Nginx might not start properly.${NC}"
        echo "You can manually kill the process with: sudo kill -9 \$(lsof -i:8080 -t)"
        read -p "Do you want to continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Set up nginx configuration
echo -e "${GREEN}Setting up nginx reverse proxy...${NC}"
# Check if the user has sudo privileges
if sudo -n true 2>/dev/null; then
    # Create a backup of the current nginx configuration
    sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup.$(date +%Y%m%d%H%M%S)
    
    # Copy our nginx configuration
    sudo cp nginx.conf /etc/nginx/conf.d/othertales-datasets.conf
    
    # Test the configuration
    if sudo nginx -t; then
        echo -e "${GREEN}Nginx configuration is valid. Restarting nginx...${NC}"
        sudo systemctl restart nginx
    else
        echo -e "${RED}Nginx configuration test failed. Aborting...${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Warning: No sudo privileges detected. Skipping nginx proxy setup.${NC}"
    echo -e "${YELLOW}To set up nginx manually, run: sudo ./setup-nginx.sh${NC}"
    echo -e "${YELLOW}Continuing without nginx proxy...${NC}"
fi

# Navigate to ui directory and install dependencies
echo -e "${GREEN}Installing UI dependencies...${NC}"
cd ui || exit
npm install

# Set environment variables for Next.js UI
export NEXT_PUBLIC_AGENT_API_URL=/api/agent
export NEXT_PUBLIC_LANGGRAPH_URL=/api/connect
export NEXT_PUBLIC_AGENT_CONFIG_URL=/api/config
export NEXT_PUBLIC_AGENT_NAME="OtherTales Datasets Agent"

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

# Check if port 3000 is in use and free it if necessary
if check_port 3000; then
    echo -e "${YELLOW}Port 3000 is already in use. Attempting to free it...${NC}"
    if ! kill_port_process 3000; then
        echo -e "${RED}Failed to free port 3000. UI server might not start properly.${NC}"
        echo "You can manually kill the process with: sudo kill -9 \$(lsof -i:3000 -t)"
        read -p "Do you want to continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Start the UI in the background
echo -e "${GREEN}Starting UI server on port 3000...${NC}"
npm run dev &
UI_PID=$!

# Go back to root directory
cd ..

# Print currently active LLM provider
echo -e "${GREEN}Using LLM Provider: ${BLUE}$LLM_PROVIDER${NC}"

# Check if port 2024 is in use and free it if necessary
if check_port 2024; then
    echo -e "${YELLOW}Port 2024 is already in use. Attempting to free it...${NC}"
    if ! kill_port_process 2024; then
        echo -e "${RED}Failed to free port 2024. API server might not start properly.${NC}"
        echo "You can manually kill the process with: sudo kill -9 \$(lsof -i:2024 -t)"
        read -p "Do you want to continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Start the Python agent API server
echo -e "${GREEN}Starting OtherTales Datasets API server on port 2024...${NC}"
python dataset_agent.py --api &
AGENT_PID=$!

echo -e "${BLUE}OtherTales Datasets is running!${NC}"
echo -e "Access the application at ${GREEN}http://localhost:8080${NC}"
echo -e "Direct UI access: ${GREEN}http://localhost:3000${NC}"
echo -e "Direct API access: ${GREEN}http://localhost:2024${NC}"
echo "Press Ctrl+C to stop all servers"

# The script is already executable at this point
# (This line was previously used to make the script executable)

# Keep the script running and capture Ctrl+C
trap "kill $UI_PID $AGENT_PID; echo -e '\n${BLUE}Shutting down servers...${NC}'; exit" INT
wait