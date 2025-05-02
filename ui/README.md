# OtherTales Datasets Agent UI

This is a chat interface for interacting with the OtherTales Datasets Agent.

## Features

- Real-time chat interface with the LangGraph-powered dataset agent
- Support for viewing tool calls and agent reasoning
- Support for interrupting agent runs
- Responsive design for desktop and mobile

## Environment Setup

The UI connects to a running LangGraph server. The following environment variables are used:

- `NEXT_PUBLIC_AGENT_NAME`: Display name for the agent
- `NEXT_PUBLIC_AGENT_API_URL`: URL for agent API
- `NEXT_PUBLIC_LANGGRAPH_URL`: URL for LangGraph connection
- `NEXT_PUBLIC_AGENT_CONFIG_URL`: URL for agent configuration

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm run start
```

## Architecture

- **Next.js 15** for the UI framework
- **React 19** for the component system
- **TailwindCSS 4** for styling
- **LangGraph API integration** for connecting to the agent

## Integration

The UI is designed to work with the LangGraph API (OpenAI-compatible endpoints):

- `/assistants/*` - Assistant management
- `/threads/*` - Thread management
- `/runs/*` - Run management
- `/store/*` - Store management

These endpoints are proxied through NGINX to the LangGraph server running on port 2024.