"""
Simple Dataset Creator Agent using LangGraph

This is a simplified version of the dataset_agent.py file to test the basic functionality.
"""

import os
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage

# Create a simple mock LLM
class MockLLM:
    """Simple mock LLM for testing."""
    
    def __init__(self, *args, **kwargs):
        pass
        
    def invoke(self, input, config=None, **kwargs):
        """Return a fixed response."""
        if "messages" in input:
            messages = input["messages"]
            query = messages[-1].content if messages else "No input"
            return {
                "messages": messages + [
                    SystemMessage(content=f"Received: {query}. This is a mock response.")
                ]
            }
        return {
            "output": "This is a mock response for testing."
        }

# Define a simple state type
class SimpleState(Dict):
    """Simple state type for the agent."""
    messages: List

# Create a simple tool
def echo_tool(input_str: str) -> str:
    """Simple echo tool for testing."""
    return f"Echo: {input_str}"

tools = [
    Tool(
        name="echo",
        func=echo_tool,
        description="Echoes back the input string."
    )
]

# Create a FastAPI app
app = FastAPI(title="Simple Dataset Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
_agent = None

def get_agent():
    """Get or create the agent instance."""
    global _agent
    if _agent is None:
        # Get the LLM
        llm = MockLLM()
        
        # Simple system prompt
        system_prompt = """
        You are a simple test agent. You help users test if the agent is working correctly.
        You have access to one tool: echo, which echoes back what you send to it.
        """
        
        # Create the agent
        _agent = create_react_agent(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt
        )
    
    return _agent

@app.post("/agent")
async def run_agent_api(request: Request):
    """API endpoint for the agent."""
    try:
        body = await request.json()
        message = body.get("message")
        thread_id = body.get("thread_id")
        
        if not message:
            raise ValueError("Message is required")
        
        agent = get_agent()
        
        # Invoke the agent
        messages = [HumanMessage(content=message)]
        result = agent.invoke({"messages": messages})
        
        # Extract the response
        response = result["messages"][-1].content if result.get("messages") else "No response"
        
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
        "provider": os.environ.get("LLM_PROVIDER", "openai"),
        "timestamp": os.environ.get("OPENAI_MODEL", "gpt-4o")
    }

if __name__ == "__main__":
    # Start the server
    port = int(os.environ.get("DATASET_AGENT_PORT", 2024))
    print(f"Starting Simple Dataset Agent API on http://localhost:{port}")
    uvicorn.run("simple_agent:app", host="0.0.0.0", port=port, reload=True)