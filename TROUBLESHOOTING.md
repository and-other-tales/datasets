# Troubleshooting

This document helps resolve common issues when using the Dataset Creator Agent.

## Known Issues and Solutions

### StateGraph.add_node() got an unexpected keyword argument 'display_name'

**Issue**: When starting the dataset agent, you see this error:
```
Error in agent API: StateGraph.add_node() got an unexpected keyword argument 'display_name'
```

**Cause**: This is due to an incompatibility with the LangGraph version being used. The original `dataset_agent.py` uses features from a newer version of LangGraph than what's installed.

**Solution**: 
1. Use the simplified `agent_standalone.py` instead:
   ```bash
   python agent_standalone.py
   ```
2. Or update the start.sh script to use the standalone agent:
   ```bash
   ./start.sh
   ```

### Port already in use

**Issue**: When starting the agent, the default port (2024) might already be in use by another process.

**Solution**: The agent now automatically finds a free port if the default port is unavailable. When running with `start.sh`, the script will detect and use the new port automatically.

## Configuration Issues

### LLM Provider Problems

**Issue**: The agent might not work properly because the LLM provider isn't configured correctly.

**Solution**:
1. Make sure you've set the proper environment variables for your chosen provider:
   ```bash
   # For example, for OpenAI:
   export LLM_PROVIDER="openai"
   export OPENAI_API_KEY="your-api-key"
   ```

2. Check the `env.example` file for the required variables for your provider.

3. If using AWS Bedrock, make sure your AWS credentials are set:
   ```bash
   export AWS_ACCESS_KEY_ID="your-key"
   export AWS_SECRET_ACCESS_KEY="your-secret"
   export AWS_REGION="your-region"
   ```

## UI Connection Issues

**Issue**: The UI is running, but can't connect to the agent API.

**Solution**:
1. Check that the agent is running with:
   ```bash
   curl http://localhost:2024/status
   ```

2. Verify the port the agent is using (if it found a different port):
   ```bash
   grep "Starting standalone agent on" /tmp/agent_output.log
   ```

3. Update the environment variable to match the actual port:
   ```bash
   export DATASET_AGENT_URL=http://localhost:XXXX/agent
   ```

4. Restart both the agent and UI if needed.

## Getting Additional Help

If you continue to encounter issues:

1. Check the logs for more detailed error information:
   ```bash
   cat /tmp/agent_output.log
   ```

2. Open an issue on our GitHub repository with details about your issue and the steps you've tried.