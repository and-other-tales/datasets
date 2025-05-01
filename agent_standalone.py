"""
Standalone Dataset Creator Agent API

This is a completely self-contained FastAPI server with no dependencies on LangGraph.
It provides a simple mock implementation of the dataset agent API.
"""

import os
import json
from typing import Dict, Any, Optional
import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create FastAPI app
app = FastAPI(title="Standalone Dataset API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Dataset Creator Agent API"}

@app.post("/agent")
async def agent(request: Request):
    """Mock agent endpoint."""
    data = await request.json()
    message = data.get("message", "No message provided")
    thread_id = data.get("thread_id")
    
    # Create mock response
    response = {
        "message": f"Received: {message}\n\nThis is a simple mock response from the standalone agent. The real agent is currently being fixed.",
        "status": "success",
        "thread_id": thread_id or "new_thread"
    }
    
    return response

@app.get("/status")
async def status():
    """Status endpoint."""
    return {
        "status": "running",
        "provider": "mock-standalone",
        "model": "mock-model",
        "timestamp": time.time(),
        "persistence": False
    }

@app.get("/info")
async def info():
    """Info endpoint (mirror of status)."""
    return await status()

@app.post("/config")
async def config(request: Request):
    """Mock config endpoint."""
    data = await request.json()
    return {
        "message": "Configuration updated (mock)",
        "config": data
    }

def find_free_port(start_port=2024, max_tries=10):
    """Find a free port starting from start_port."""
    import socket
    from contextlib import closing
    
    for port in range(start_port, start_port + max_tries):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            # Setting sock.settimeout prevents hanging on occupied ports
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            if result != 0:  # Port is available
                return port
    
    # If we get here, just return a high random port
    import random
    return random.randint(8000, 9000)

if __name__ == "__main__":
    # Get port from environment variable or find a free port
    preferred_port = int(os.environ.get("DATASET_AGENT_PORT", 2024))
    port = find_free_port(start_port=preferred_port)
    
    # If we found a different port, update the environment variable
    if port != preferred_port:
        os.environ["DATASET_AGENT_PORT"] = str(port)
        print(f"Port {preferred_port} was not available. Using port {port} instead.")
    
    print(f"Starting standalone agent on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)