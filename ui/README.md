# Dataset Creator Chat UI

A Next.js-based chat interface for the HuggingFace Dataset Creator Agent. This UI allows users to interact with the agent through a modern, responsive chat interface.

## Features

- Real-time chat interface with the Dataset Creator Agent
- Markdown support for rich text responses
- Code highlighting for code blocks
- Message history with auto-scrolling
- Responsive design that works on mobile and desktop
- Optimized for Google Cloud Run deployment

## Getting Started

### Prerequisites

- Node.js 22+ 
- npm or yarn or pnpm

### Installation

1. Clone the repository or navigate to this folder

```bash
cd ui
```

2. Install dependencies

```bash
npm install
# or
yarn install
# or
pnpm install
```

3. Start the development server

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser to see the chat UI

## Environment Variables

Copy the `.env.example` file to `.env.local` and update the values as needed:

```
# Agent configuration
NEXT_PUBLIC_AGENT_NAME=Dataset Creator Agent
AGENT_API_URL=http://localhost:8000

# Storage configuration (used in production)
STORAGE_DIR=/gcs/dataset-creator

# Server configuration
PORT=8080
NODE_ENV=development
```

## Integration with the Dataset Creator Agent

This UI is designed to work with the Dataset Creator Agent, a Python-based agent that creates HuggingFace datasets from web content. The agent is powered by AWS Bedrock's Claude 3 model and uses LangGraph for orchestration.

### How it Works

1. The user sends a message through the chat interface
2. The message is sent to the agent via the AgentClient
3. The agent processes the message and returns a response
4. The response is displayed in the chat interface

## Deployment

### Google Cloud Run Deployment

This application is optimized to run on Google Cloud Run. The following steps outline how to deploy it:

#### Prerequisites

- Google Cloud SDK installed and configured
- Docker installed
- A Google Cloud project with Cloud Run API enabled
- A Google Cloud Storage bucket for data storage

#### Manual Deployment

1. Build the Docker image:

```bash
docker build -t gcr.io/[YOUR_PROJECT_ID]/dataset-creator-chat:latest .
```

2. Push the image to Google Container Registry:

```bash
docker push gcr.io/[YOUR_PROJECT_ID]/dataset-creator-chat:latest
```

3. Deploy to Cloud Run:

```bash
gcloud run deploy dataset-creator-chat \
  --image gcr.io/[YOUR_PROJECT_ID]/dataset-creator-chat:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 1 \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 10 \
  --set-env-vars=NODE_ENV=production,NEXT_PUBLIC_AGENT_NAME="Dataset Creator Agent",AGENT_API_URL=https://your-agent-api.com \
  --volume=name=gcs,mount-path=/gcs \
  --volume-type=gcs
```

#### Continuous Deployment with Cloud Build

This repository includes a `cloudbuild.yaml` file that can be used for continuous deployment:

1. Connect your repository to Cloud Build
2. Set up a trigger that uses the `cloudbuild.yaml` file
3. Configure the necessary environment variables in the Cloud Build trigger

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.