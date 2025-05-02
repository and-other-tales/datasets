# OtherTales Dataset Creator

A powerful tool for creating high-quality datasets for LLM training using intelligent web crawling and content processing. This tool helps developers build custom datasets compatible with HuggingFace Datasets, making it easy to create specialized training data for various domains.

## Features

- 🕷️ **Intelligent Web Crawling**: Recursively crawl websites with configurable depth and pattern matching
- 🔄 **Content Processing**: Advanced HTML to Markdown conversion with content cleaning
- 📊 **Dataset Creation**: Automatic structuring of crawled content into HuggingFace datasets
- 🤖 **Multi-Provider AI Support**: Compatible with major LLM providers:
  - OpenAI (GPT-4.1, 4o, o1, o3-mini)
  - Anthropic (Claude 3.7 & 3.6 & Sonnet)
  - AWS Bedrock (Amazon Titan & 3rd Party Provider Models - Anthropic / Llama)
  - HuggingFace (Llama 4, 3, HF Transformer Models)
  - Groq
  - Google Vertex AI (Gemini)
- 📈 **Observability**: Built-in tracing with LangSmith for monitoring and debugging
- 🔍 **Verification**: Tools to verify and validate created datasets
- ☁️ **HuggingFace Integration**: Direct upload to HuggingFace Hub

## Installation

```bash
git clone https://github.com/and-other-tales/datasets.git
cd datasets
pip install -r requirements.txt
```

## Environment Setup

### Required Environment Variables

See the `env.example` file for a complete list of environment variables. The basic configuration is:

```bash
# LLM Provider Configuration
# Set which LLM provider to use
export LLM_PROVIDER="bedrock"  # Options: openai, anthropic, bedrock, azure, google, groq, huggingface

# LangChain Configuration
export LANGCHAIN_API_KEY="your_langchain_api_key"

# Provider-specific variables (depending on which provider you chose)

# 1. AWS Bedrock Configuration (default)
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export AWS_REGION="us-west-2"
export BEDROCK_MODEL_ID="anthropic.claude-3-7-sonnet-20250219-v1:0"

# 2. OpenAI Configuration
# export OPENAI_API_KEY="your_openai_key"
# export OPENAI_MODEL="gpt-4o"  # or gpt-3.5-turbo

# 3. Anthropic Configuration
# export ANTHROPIC_API_KEY="your_anthropic_key"
# export ANTHROPIC_MODEL="claude-3-7-sonnet-latest"

# 4. Azure OpenAI Configuration
# export AZURE_OPENAI_API_KEY="your_azure_openai_key_here"
# export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com"
# export AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name"

# 5. Google Generative AI Configuration
# export GOOGLE_API_KEY="your_google_api_key_here"
# export GOOGLE_MODEL="gemini-1.5-pro"

# 6. Groq Configuration
# export GROQ_API_KEY="your_groq_api_key_here"
# export GROQ_MODEL="llama3-70b-8192"

# 7. HuggingFace Configuration
# export HUGGINGFACE_API_KEY="your_huggingface_api_key_here"
# export HUGGINGFACE_MODEL_ID="mistralai/Mixtral-8x7B-Instruct-v0.1"
```

You can find a more detailed list of available environment variables in the `env.example` file.

## Usage

### Basic Example

```python
from datasets import build_agent

# Initialize the agent with default provider (from env)
agent = build_agent()

# Or specify provider explicitly
agent = build_agent(provider="openai", model="gpt-4")

# Create a dataset from a website
response = agent.invoke({
    "input": "Create a dataset from https://example.com/docs"
})
```

### Advanced Example: Creating a Domain-Specific Dataset

```python
# Example: Creating a UK Tax Advisory Dataset
response = agent.invoke({
    "input": """
    Create a dataset from https://www.gov.uk/government/collections/hmrc-manuals
    that contains HMRC guidance and manuals for tax advisers.
    
    Parameters:
    - max_depth: 3
    - patterns_to_match: ["*/manual/*", "*/guidance/*"]
    - exclude: ["*/archives/*"]
    """
})
```

## Dataset Structure

Created datasets include the following columns:
- `url`: Source URL of the content
- `title`: Page title or document heading
- `text`: Clean, processed text content
- `html`: Original HTML content (optional)
- `metadata`: Additional metadata including:
  - `crawl_timestamp`
  - `last_modified`
  - `section_headers`
  - `categories`

## Advanced Configuration

### Crawling Configuration

```python
{
    "url": "https://example.com",
    "max_depth": 2,
    "max_pages": 50,
    "patterns_to_match": ["*/docs/*", "*/guide/*"],
    "patterns_to_exclude": ["*/archived/*", "*/deprecated/*"]
}
```

### Dataset Creation Options

```python
{
    "dataset_name": "my_custom_dataset",
    "push_to_hub": True,
    "hub_username": "your_username",
    "dataset_description": "A custom dataset for..."
}
```

### Model Provider Configuration

You can configure the LLM provider in several ways:

#### 1. Using Environment Variables

```python
# Configuration via environment variables
import os

# Set provider and model
os.environ["LLM_PROVIDER"] = "anthropic"
os.environ["ANTHROPIC_MODEL"] = "claude-3-7-sonnet-latest"
os.environ["ANTHROPIC_TEMPERATURE"] = "0.2"
```

#### 2. Using the API Config Endpoint

You can change the LLM provider at runtime using the `/config` API endpoint:

```python
import requests

# Change to OpenAI
requests.post("http://localhost:8080/config", json={
    "llm_provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.2
})

# Change to Anthropic
requests.post("http://localhost:8080/config", json={
    "llm_provider": "anthropic",
    "model": "claude-3-7-sonnet-latest",
    "temperature": 0.2
})

# Change to AWS Bedrock
requests.post("http://localhost:8080/config", json={
    "llm_provider": "bedrock",
    "model": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "temperature": 0.2
})
```

#### 3. From the Web UI (Coming Soon)

   A provider selection dropdown will be available in the web UI in future releases.

## LangGraph CLI & Development Server

This project includes a [LangGraph CLI](https://langchain-ai.github.io/langgraph) development server for local testing and rapid iteration.

### Local Development

1. Install the CLI with in-memory support:

```bash
pip install -U "langgraph-cli[inmem]"
```

2. Copy the environment file and fill in your values:

```bash
cp env.example .env
```

3. Start the development server:

```bash
langgraph dev
```

The server will start on http://localhost:2024 by default, serving all configured graphs.

### Docker

Build the Docker image:

```bash
docker build -t othertales-datasets .
```

Run the container:

```bash
docker run --env-file .env -p 2024:2024 othertales-datasets
```

### Docker Compose

Alternatively, use Docker Compose:

```bash
docker compose up
```

Verify the service is running:

```bash
curl http://localhost:2024/ok
```

## Use Cases

1. **Regulatory Compliance Training**
   - Create datasets from government regulatory documents
   - Train models on specific compliance guidelines
   - Example: HMRC tax guidance, FSA regulations

2. **Legal Knowledge Bases**
   - Build datasets from legislation and legal documents
   - Create specialized legal assistant models
   - Example: UK legislation, case law databases

3. **Technical Documentation**
   - Create datasets from software documentation
   - Train models for technical support
   - Example: API documentation, user guides

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Supports multiple LLM providers:
  - OpenAI
  - Anthropic
  - AWS Bedrock
  - HuggingFace
  - Groq
  - Google Vertex AI
- Uses [HuggingFace Datasets](https://github.com/huggingface/datasets)

## Support

For support, please [open an issue](https://github.com/and-other-tales/datasets/issues) on GitHub.
