#!/bin/bash

# Setup script for nginx reverse proxy configuration

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up nginx reverse proxy for OtherTales Datasets...${NC}"

# Check if nginx is installed
if ! command -v nginx &> /dev/null; then
    echo -e "${GREEN}Installing nginx...${NC}"
    sudo apt-get update
    sudo apt-get install -y nginx
fi

# Create a backup of the current nginx configuration
echo -e "${GREEN}Backing up current nginx configuration...${NC}"
sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup

# Copy our nginx configuration
echo -e "${GREEN}Copying new nginx configuration...${NC}"
sudo cp nginx.conf /etc/nginx/conf.d/othertales-datasets.conf

# Test the configuration
echo -e "${GREEN}Testing nginx configuration...${NC}"
sudo nginx -t

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Configuration is valid. Restarting nginx...${NC}"
    sudo systemctl restart nginx
    echo -e "${BLUE}Nginx reverse proxy setup complete!${NC}"
    echo -e "Now you can access:"
    echo -e "  - Web UI: http://localhost:8080/"
    echo -e "  - API endpoints via http://localhost:8080/assistants/* and other paths"
else
    echo -e "\nNginx configuration test failed. Please check the configuration file."
    echo -e "Restoring backup configuration..."
    sudo cp /etc/nginx/nginx.conf.backup /etc/nginx/nginx.conf
fi