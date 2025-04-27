# HuggingFace Dataset Creator Agent

This agent creates HuggingFace datasets from URLs by crawling, downloading, and processing content.

## Features

- URL crawling with configurable depth and filters using Playwright
- HTML to Markdown conversion using BeautifulSoup4 and JinaAI/ReaderLM-v2 approach
- HuggingFace dataset creation with proper metadata
- Complete workflow: User Prompt → URL Crawl & Download → HTML Conversion → Dataset Generation
- Chat interface with AWS Bedrock (Claude 3.7 Sonnet)

## Workflow

1. **User Prompt Task**: The user provides a URL or website to crawl for dataset creation
2. **URL Crawl & Download**: The agent crawls the website using Playwright, respecting depth limits and URL patterns
3. **HTML Conversion**: HTML is cleaned and converted to high-quality markdown using the ReaderLM approach
4. **Dataset Generation**: A structured HuggingFace dataset is created, with options to push to the Hub

## Requirements

- Python 3.8+
- beautifulsoup4
- playwright
- langchain
- langgraph
- datasets
- AWS Bedrock credentials

## Installation

```bash
# Install dependencies
pip install langchain-aws langgraph playwright beautifulsoup4 datasets lxml

# Install Playwright browsers
playwright install
```

## Usage

```bash
# Set your AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=your_region

# Run the agent
python datasets.py
```

## Example

```
Your request: Create a dataset from https://www.gov.uk/government/collections/hmrc-manuals that contains the entire HMRC Manuals collection.

Agent: I'll help you create a dataset from the HMRC Manuals collection. Let's break this down into steps:

1. First, I'll crawl the website to gather the content
2. Then create a HuggingFace dataset from the crawled content
3. Finally, verify the dataset was created successfully

Let me start by crawling the URL...

[Crawling progress shown here]

I've completed crawling the HMRC Manuals collection. I found 87 pages with valid content including various tax manuals.

Now creating the dataset 'hmrc_manuals'...

Successfully created dataset 'hmrc_manuals' with 87 documents.
Dataset contains 87 entries with columns: ['url', 'title', 'text', 'html', 'metadata']

The dataset has been created successfully and is ready for use. Would you like me to push this to the HuggingFace Hub or make any adjustments to the dataset?
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.