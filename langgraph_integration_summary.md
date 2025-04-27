# LangGraph Integration Summary

This document provides a comprehensive overview of key concepts and guidelines for integrating LangGraph into our codebase.

## 1. Working with LangGraph Studio

LangGraph Studio is a visual IDE for developing and iterating on your LangGraph applications:

- **Prompt Engineering**: Define special configuration fields for prompts using `langgraph_nodes` and `langgraph_type` metadata in your Pydantic models or dataclasses
- **Configuration Visibility**: Add configuration icons to nodes by specifying which fields are associated with each node
- **Playground Integration**: Test and iterate on prompts through the LangSmith Playground without re-running the entire graph
- **Node Visualization**: View LLM runs, inputs/outputs, and execution paths through the graph UI

Example configuration:
```python
class Configuration(BaseModel):
    """The configuration for the agent."""
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="The system prompt to use for the agent's interactions.",
        json_schema_extra={
            "langgraph_nodes": ["call_model"],
            "langgraph_type": "prompt",
        },
    )
```

## 2. Configuration Best Practices

Runtime configuration allows customization of your graph at runtime:

- **Configurable Parameters**: Pass in a `configurable` dictionary in the config parameter when calling `invoke()`
- **Config Schema**: Define a TypedDict or Pydantic model to specify the configuration options
- **Access Configuration**: In node functions, access config through the `config["configurable"]` dictionary
- **Common Use Cases**: LLM selection, system messages, behavior flags

Example:
```python
def _call_model(state: AgentState, config: RunnableConfig):
    # Access the config through the configurable key
    model_name = config["configurable"].get("model", "default_model")
    system_message = config["configurable"].get("system_message", "Default system message")
    
    # Use the configuration
    model = models[model_name]
    messages = state["messages"]
    if system_message:
        messages = [SystemMessage(content=system_message)] + messages
    
    response = model.invoke(messages)
    return {"messages": [response]}
```

## 3. CLI Commands for Development

The LangGraph CLI provides tools for local development and deployment:

- **Installation**: `pip install langgraph-cli` or `npm install -g @langchain/langgraph-cli`
- **Configuration**: Create a `langgraph.json` file that defines dependencies, graphs, auth, and environment settings
- **Key Config Properties**:
  - `dependencies`: Array of Python package dependencies
  - `graphs`: Mapping of graph IDs to paths where graphs are defined
  - `auth`: Optional authentication configuration
  - `store`: Configuration for semantic search and TTL settings
  - `checkpointer`: Configuration for checkpoint expiration
  - `http`: HTTP server configuration including CORS settings

Example configuration:
```json
{
  "dependencies": ["."],
  "graphs": {
    "chat": "./chat/graph.py:graph"
  },
  "store": {
    "index": {
      "embed": "openai:text-embedding-3-small",
      "dims": 1536,
      "fields": ["$"]
    },
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 60,
      "default_ttl": 10080
    }
  }
}
```

## 4. Graph Structures and Types

LangGraph provides flexible graph structures for building agent workflows:

- **Graph Classes**: 
  - `Graph`: Base graph class for defining workflows
  - `CompiledGraph`: Executable graph after compilation
  
- **Key Methods**:
  - `add_conditional_edges(source, path, path_map, then)`: Add conditional transitions between nodes
  - `set_entry_point(key)`: Define the starting node (equivalent to `add_edge(START, key)`)
  - `set_conditional_entry_point(path, path_map, then)`: Define a dynamic starting point
  - `set_finish_point(key)`: Mark a node where graph execution should end

- **Execution Flow**:
  - Graphs execute nodes in sequence based on edges
  - Conditional edges determine the next node dynamically based on state
  - Special constants `START` and `END` define entry and exit points

Example:
```python
builder = StateGraph(AgentState)
builder.add_node("model", _call_model)
builder.add_edge(START, "model")
builder.add_conditional_edges(
    "model",
    path=_router,
    path_map={"tool": "use_tool", "response": "generate_response"}
)
builder.add_edge("use_tool", "model")
builder.add_edge("generate_response", END)
graph = builder.compile()
```

## 5. Checkpoint Persistence

Checkpointers allow LangGraph agents to persist state within and across interactions:

- **Checkpoint Components**:
  - `Checkpoint`: State snapshot at a specific point in time
  - `CheckpointMetadata`: Additional information about the checkpoint
  - `BaseCheckpointSaver`: Base class for implementing custom persistence

- **Key Methods**:
  - `get(config)`: Fetch a checkpoint using configuration
  - `put(config, checkpoint, metadata, new_versions)`: Store a checkpoint
  - `list(config, filter, before, limit)`: List checkpoints matching criteria
  - `delete_thread(thread_id)`: Remove all checkpoints for a thread

- **TTL Configuration**:
  - Configure checkpoint expiration with `strategy`, `sweep_interval_minutes`, and `default_ttl`
  - Example: `{"strategy": "delete", "sweep_interval_minutes": 60, "default_ttl": 43200}`

Example:
```python
checkpointer = PostgresSaver(pool)
checkpointer.setup()  # Must be called once before use
graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
```

## 6. Agent Building Blocks

LangGraph provides prebuilt components for common agent behaviors:

- **Core Packages**:
  - `langgraph-prebuilt`: Base components for creating agents
  - `langgraph-supervisor`: Tools for supervisor agents
  - `langgraph-swarm`: Tools for multi-agent systems
  - `langchain-mcp-adapters`: Interfaces to MCP servers
  - `langmem`: Agent memory management (short and long-term)
  - `agentevals`: Evaluation utilities

- **Key Features**:
  - Memory integration (session-based and persistent)
  - Human-in-the-loop control with indefinite pausing
  - Streaming support for real-time updates
  - Deployment and infrastructure tools

## 7. Postgres Persistence Integration

LangGraph supports PostgreSQL for persistent state storage:

- **Setup**:
  1. Install required packages: `pip install -U psycopg psycopg-pool langgraph-checkpoint-postgres`
  2. Define database connection URI and parameters
  3. Create a PostgreSQL saver (synchronous or asynchronous)
  4. Call `.setup()` to initialize the database schema
  5. Pass the checkpointer to your graph during compilation

- **Connection Options**:
  - **Connection Pool**: Efficient for applications with many short-lived operations
  - **Sync vs Async**: Choose based on your application's concurrency requirements

Example:
```python
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://username:password@localhost:5432/dbname"
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

with ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs=connection_kwargs,
) as pool:
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()  # Initialize schema
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    
    # Use thread_id to maintain conversation state
    config = {"configurable": {"thread_id": "user123"}}
    response = graph.invoke({"messages": [("human", "Hello")]}, config=config)
```

## Conclusion

LangGraph provides a comprehensive framework for building, testing, and deploying agent systems with a focus on state management, flexibility, and production readiness. By leveraging these components, you can rapidly develop complex agent workflows while maintaining control over their behavior and persistence.