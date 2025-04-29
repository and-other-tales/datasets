# Changes for Google Cloud Run Deployment

This document outlines the changes made to enable the application to run on Google Cloud Run with a single exposed port (8080).

## Overview

The application has been refactored to use nginx as a reverse proxy, routing requests to:
- Next.js UI (port 3000)
- LangGraph server (port 2024)

## Key Changes

1. **nginx Configuration**
   - Updated `nginx.conf` to route requests based on path:
     - UI requests (`/`) → Next.js on port 3000
     - Next.js API requests (`/api/*`) → Next.js on port 3000
     - LangGraph requests (`/agent`, `/status`, `/assistants/*`, etc.) → LangGraph on port 2024

2. **LangGraph Configuration**
   - Updated `langgraph.json` to use port 2024 instead of 8000
   - `langgraph dev` command is used to start the LangGraph server

3. **Docker Integration**
   - Consolidated into a single Dockerfile with nginx, Next.js, and Python
   - Created `cloudrun-start.sh` to launch all services in the right order
   - Removed `Dockerfile.nginx` as it's now integrated in the main Dockerfile

4. **API Routes**
   - Updated API route configurations in Next.js to work with the reverse proxy
   - Changed endpoint URLs to use the nginx routes

5. **Documentation**
   - Updated README with deployment instructions
   - Added details about the nginx proxy setup

## How It Works

The startup process in Cloud Run:
1. Nginx starts and binds to port 8080 (the exposed port)
2. Next.js starts on port 3000 (internal)
3. LangGraph server starts on port 2024 (internal)
4. All incoming requests go through nginx, which routes them to the appropriate service

## Testing

To test locally:
```bash
# Build and run with docker-compose
docker-compose up --build

# Test via http://localhost:8080
```

## Deployment to Cloud Run

Deploy using the Cloud Build configuration:
```bash
gcloud builds submit --config cloudbuild.yaml
```