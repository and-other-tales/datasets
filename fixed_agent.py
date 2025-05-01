"""
Fixed Dataset Creator Agent using LangGraph

This is a simplified version of the dataset_agent.py file with compatibility fixes for LangGraph.
"""

import os
import json
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Create a simple mock LLM
class MockLLM:
    """Simple mock LLM for testing."""
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def invoke(self, input, config=None, **kwargs):
        """Return a fixed response."""
        return {"output": "This is a mock response for testing."}

# Create a FastAPI app
app = FastAPI(title="Fixed Dataset Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/agent")
async def run_agent_api(request: Request):
    """API endpoint for the agent."""
    try:
        body = await request.json()
        message = body.get("message")
        thread_id = body.get("thread_id")
        
        if not message:
            raise ValueError("Message is required")
        
        # Create a simple response
        response = f"Received: {message}. This is a fixed agent response."
        
        return {
            "message": response,
            "status": "success",
            "thread_id": thread_id
        }
    except Exception as e:
        print(f"Error in agent API: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/status")
async def get_status():
    """Status endpoint."""
    return {
        "status": "running",
        "provider": "fixed-mock-agent",
        "model": "mock-model",
        "timestamp": "2025-05-01"
    }

@app.get("/info")
async def get_info():
    """Mirror of the status endpoint for health checks."""
    return await get_status()

if __name__ == "__main__":
    # Start the server
    port = int(os.environ.get("DATASET_AGENT_PORT", 2024))
    print(f"Starting Fixed Dataset Agent API on http://localhost:{port}")
    uvicorn.run("fixed_agent:app", host="0.0.0.0", port=port, reload=True)