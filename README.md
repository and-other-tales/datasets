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
```bash
# LangChain Configuration
export LANGCHAIN_API_KEY="your_langchain_api_key"

# Model Provider Configuration (choose one or more)
export MODEL_PROVIDER="openai"  # Options: openai, anthropic, bedrock, huggingface, groq, vertex

# OpenAI Configuration
export OPENAI_API_KEY="your_openai_key"
export OPENAI_MODEL="gpt-4"  # or gpt-3.5-turbo

# Anthropic Configuration
export ANTHROPIC_API_KEY="your_anthropic_key"
export ANTHROPIC_MODEL="claude-3-opus"  # or claude-3-sonnet

# AWS Bedrock Configuration
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export BEDROCK_MODEL="anthropic.claude-v2"  # or amazon.titan-text

# HuggingFace Configuration
export HUGGINGFACE_TOKEN="your_huggingface_token"
export HF_MODEL="mistralai/mixtral-8x7b"

# Groq Configuration
export GROQ_API_KEY="your_groq_key"
export GROQ_MODEL="mixtral-8x7b"

# Google Vertex AI Configuration
export VERTEX_PROJECT="your_project_id"
export VERTEX_LOCATION="your_location"
export VERTEX_MODEL="gemini-pro"
```

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

```python
# Configuration via environment variables
import os

# Set provider and model
os.environ["MODEL_PROVIDER"] = "anthropic"
os.environ["ANTHROPIC_MODEL"] = "claude-3-opus"

# Or configure multiple providers for fallback
providers = {
    "primary": {
        "provider": "openai",
        "model": "gpt-4",
    },
    "fallback": {
        "provider": "anthropic",
        "model": "claude-3-opus",
    }
}

agent = build_agent(providers=providers)
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
