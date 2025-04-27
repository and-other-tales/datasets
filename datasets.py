"""
HuggingFace Dataset Creator Agent

This agent creates HuggingFace datasets from URLs by crawling, downloading, and processing content.

Workflow:
1. User Prompt Task > 
2. URL Crawl & Download > 
3. HTML Conversion > 
4. Dataset Generation
"""

import os
import re
import json
import asyncio
from typing import List, Dict, Any, Optional, Union, Literal
import hashlib
import tempfile
from pathlib import Path
import time
from urllib.parse import urljoin, urlparse

# Web scraping and processing
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import markdown

# LangGraph and LangChain components
from langgraph.graph import MessageGraph, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool, StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field

# AWS Bedrock for LLM
from langchain_aws import ChatBedrockConverse

# For dataset creation
from datasets import Dataset, DatasetDict, Features, Value, Sequence
import datasets

# Configuration 
MODEL_ID = "anthropic.claude-3-7-sonnet-20250219-v1:0"
MAX_DEPTH = 3  # Maximum crawling depth
MAX_PAGES = 100  # Maximum number of pages to crawl per domain
TIMEOUT = 30000  # Timeout for page loading in ms

# Set up directories
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dataset_crawler")
os.makedirs(CACHE_DIR, exist_ok=True)

class UrlCrawlInput(BaseModel):
    """Input for the URL crawler."""
    url: str = Field(..., description="The URL to crawl")
    max_depth: int = Field(default=2, description="Maximum crawl depth")
    max_pages: int = Field(default=50, description="Maximum number of pages to crawl")
    patterns_to_match: Optional[List[str]] = Field(default=None, 
        description="List of URL patterns to include (regex patterns)")
    patterns_to_exclude: Optional[List[str]] = Field(default=None,
        description="List of URL patterns to exclude (regex patterns)")
    
class DatasetCreationInput(BaseModel):
    """Input for dataset creation."""
    dataset_name: str = Field(..., description="Name for the HuggingFace dataset")
    dataset_files: List[Dict[str, str]] = Field(..., 
        description="List of files with 'url' and 'content' keys")
    push_to_hub: bool = Field(default=False, 
        description="Whether to push the dataset to HuggingFace Hub")
    hub_username: Optional[str] = Field(default=None,
        description="HuggingFace Hub username")
    dataset_description: Optional[str] = Field(default=None,
        description="Description of the dataset")

def clean_html(html_content: str) -> str:
    """Clean HTML content by removing scripts, styles, and comments."""
    # Patterns
    SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
    STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
    META_PATTERN = r"<[ ]*meta.*?>"
    COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
    LINK_PATTERN = r"<[ ]*link.*?>"
    BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'

    # Clean HTML
    html_content = re.sub(SCRIPT_PATTERN, "", html_content, 
                        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html_content = re.sub(STYLE_PATTERN, "", html_content, 
                        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html_content = re.sub(META_PATTERN, "", html_content, 
                        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html_content = re.sub(COMMENT_PATTERN, "", html_content, 
                        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html_content = re.sub(LINK_PATTERN, "", html_content, 
                        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html_content = re.sub(BASE64_IMG_PATTERN, "", html_content, 
                        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    return html_content

def html_to_markdown(html_content: str) -> str:
    """
    Convert HTML content to markdown using ReaderLM pattern.
    This implementation follows the ReaderLM method described in REFERENCE/MD.txt
    """
    # Clean the HTML first
    html_content = clean_html(html_content)
    
    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Create a more structured version for processing
    # Process headings properly
    for i in range(1, 7):
        for heading in soup.find_all(f'h{i}'):
            heading.insert_before(BeautifulSoup(f'\n\n{"#" * i} ', 'lxml'))
    
    # Process lists properly
    for ul in soup.find_all('ul'):
        for li in ul.find_all('li'):
            li.insert_before(BeautifulSoup('\n* ', 'lxml'))
    
    for ol in soup.find_all('ol'):
        for i, li in enumerate(ol.find_all('li')):
            li.insert_before(BeautifulSoup(f'\n{i+1}. ', 'lxml'))
    
    # Process paragraphs
    for p in soup.find_all('p'):
        p.insert_before(BeautifulSoup('\n\n', 'lxml'))
        p.append(BeautifulSoup('\n', 'lxml'))
    
    # Process links
    for a in soup.find_all('a', href=True):
        text = a.get_text()
        href = a['href']
        a.replace_with(BeautifulSoup(f'[{text}]({href})', 'lxml'))
    
    # Process images
    for img in soup.find_all('img', src=True):
        alt = img.get('alt', '')
        src = img['src']
        img.replace_with(BeautifulSoup(f'![{alt}]({src})', 'lxml'))
    
    # Get the text and clean it up
    text = soup.get_text()
    
    # Clean up extra newlines and whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

def get_page_metadata(html_content: str, url: str) -> Dict[str, Any]:
    """Extract metadata from an HTML page."""
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Extract basic metadata
    title = soup.title.string if soup.title else ""
    
    # Extract meta tags
    meta_description = ""
    meta_keywords = ""
    
    for meta in soup.find_all('meta'):
        if meta.get('name', '').lower() == 'description':
            meta_description = meta.get('content', '')
        if meta.get('name', '').lower() == 'keywords':
            meta_keywords = meta.get('content', '')
    
    return {
        "url": url,
        "title": title,
        "description": meta_description,
        "keywords": meta_keywords,
        "crawled_at": time.time()
    }

def get_cache_path(url: str) -> Path:
    """Get path for cached content."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return Path(CACHE_DIR) / f"{url_hash}.json"

def is_cached(url: str) -> bool:
    """Check if URL content is cached."""
    cache_path = get_cache_path(url)
    return cache_path.exists()

def get_cached_content(url: str) -> Dict[str, Any]:
    """Get cached content for a URL."""
    cache_path = get_cache_path(url)
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

def save_to_cache(url: str, content: Dict[str, Any]) -> None:
    """Save content to cache."""
    cache_path = get_cache_path(url)
    with open(cache_path, 'w') as f:
        json.dump(content, f)

async def process_url(url: str) -> Dict[str, Any]:
    """Process a single URL to get content and metadata."""
    # Check cache first
    if is_cached(url):
        return get_cached_content(url)
    
    # If not cached, fetch and process
    try:
        # Use playwright to get rendered HTML
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, timeout=TIMEOUT)
            html_content = await page.content()
            await browser.close()
            
        # Get metadata
        metadata = get_page_metadata(html_content, url)
        
        # Convert to markdown
        markdown_content = html_to_markdown(html_content)
        
        # Create result
        result = {
            "url": url,
            "html_content": html_content,
            "markdown_content": markdown_content,
            "metadata": metadata,
        }
        
        # Cache the result
        save_to_cache(url, result)
        
        return result
    
    except Exception as e:
        # Handle the error gracefully
        error_content = {
            "url": url,
            "error": str(e),
            "metadata": {"url": url, "error": str(e), "crawled_at": time.time()}
        }
        save_to_cache(url, error_content)
        return error_content

async def recursive_crawl(start_url: str, max_depth: int = MAX_DEPTH, 
                      max_pages: int = MAX_PAGES, 
                      patterns_to_match: List[str] = None,
                      patterns_to_exclude: List[str] = None) -> List[Dict[str, Any]]:
    """Recursively crawl URLs starting from a base URL."""
    visited = set()
    to_visit = [(start_url, 0)]  # (url, depth)
    results = []
    
    # Compile regex patterns
    if patterns_to_match:
        include_patterns = [re.compile(pattern) for pattern in patterns_to_match]
    else:
        include_patterns = []
    
    if patterns_to_exclude:
        exclude_patterns = [re.compile(pattern) for pattern in patterns_to_exclude]
    else:
        exclude_patterns = []
    
    # Get the base domain for relative URL resolution
    base_domain = urlparse(start_url).netloc
    
    while to_visit and len(visited) < max_pages:
        url, depth = to_visit.pop(0)
        
        if url in visited:
            continue
        
        if depth > max_depth:
            continue
        
        # Mark as visited before processing
        visited.add(url)
        
        # Process URL
        result = await process_url(url)
        results.append(result)
        print(f"Processed {url} (depth {depth})")
        
        # If we hit max pages, stop
        if len(visited) >= max_pages:
            break
        
        # Extract links and add to queue if within the same domain
        if result.get("html_content"):
            soup = BeautifulSoup(result["html_content"], 'lxml')
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link['href']
                # Resolve relative URLs
                full_url = urljoin(url, href)
                parsed_url = urlparse(full_url)
                
                # Skip if not HTTP(S)
                if not parsed_url.scheme.startswith('http'):
                    continue
                
                # Skip if different domain
                if parsed_url.netloc != base_domain:
                    continue
                
                # Skip if already visited or in queue
                if full_url in visited:
                    continue
                
                # Skip if fragment or query only changes
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                if base_url in visited:
                    continue
                
                # Check against include/exclude patterns
                should_include = True
                
                # Check include patterns (if any)
                if include_patterns:
                    should_include = any(pattern.search(full_url) for pattern in include_patterns)
                
                # Check exclude patterns (if any)
                if exclude_patterns:
                    if any(pattern.search(full_url) for pattern in exclude_patterns):
                        should_include = False
                
                if should_include:
                    to_visit.append((full_url, depth + 1))
    
    return results

async def batch_crawl(urls: List[str]) -> List[Dict[str, Any]]:
    """Crawl multiple URLs concurrently."""
    tasks = [process_url(url) for url in urls]
    return await asyncio.gather(*tasks)

def filter_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter results to remove errors and empty content."""
    return [r for r in results if "error" not in r and r.get("markdown_content")]

def create_huggingface_dataset(
    dataset_name: str, 
    dataset_files: List[Dict[str, Any]],
    push_to_hub: bool = False, 
    hub_username: str = None, 
    dataset_description: str = None
) -> Dataset:
    """Create a HuggingFace dataset from crawled files."""
    # Prepare the data
    data = {
        "url": [],
        "title": [],
        "text": [],
        "html": [],
        "metadata": []
    }
    
    for item in dataset_files:
        data["url"].append(item["url"])
        data["title"].append(item["metadata"].get("title", ""))
        data["text"].append(item["markdown_content"])
        data["html"].append(item["html_content"])
        data["metadata"].append(json.dumps(item["metadata"]))
    
    # Create the dataset
    dataset = Dataset.from_dict(data)
    
    # Create DatasetDict with a single split
    dataset_dict = DatasetDict({"train": dataset})
    
    # If pushing to hub
    if push_to_hub:
        if not hub_username:
            raise ValueError("hub_username is required when push_to_hub is True")
        
        repo_name = f"{hub_username}/{dataset_name}"
        dataset_dict.push_to_hub(
            repo_name,
            private=True,
            token=os.environ.get("HF_TOKEN"),
            commit_message=f"Create dataset {dataset_name}",
            readme_dict={
                "dataset_info": {
                    "description": dataset_description or f"Web crawled dataset from {dataset_name}",
                    "license": "other",
                    "citation": ""
                }
            }
        )
        
        print(f"Dataset pushed to hub: {repo_name}")
    
    return dataset_dict

# Tool definitions for the agent
async def crawl_url_tool(input_data: Union[str, Dict]) -> str:
    """Tool to crawl URLs and extract content."""
    if isinstance(input_data, str):
        # Parse as JSON if it's a string
        try:
            data = json.loads(input_data)
        except json.JSONDecodeError:
            # If not valid JSON, assume it's just the URL
            data = {"url": input_data}
    else:
        data = input_data
    
    url = data["url"]
    max_depth = data.get("max_depth", 2)
    max_pages = data.get("max_pages", 50)
    patterns_to_match = data.get("patterns_to_match", None)
    patterns_to_exclude = data.get("patterns_to_exclude", None)
    
    results = await recursive_crawl(
        url, 
        max_depth=max_depth, 
        max_pages=max_pages,
        patterns_to_match=patterns_to_match,
        patterns_to_exclude=patterns_to_exclude
    )
    
    # Filter results and return summary
    filtered_results = filter_results(results)
    
    # Cache the results for dataset creation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(filtered_results, f)
        temp_path = f.name
    
    return f"""
Crawled {url} with depth {max_depth}. 
Found {len(results)} pages, {len(filtered_results)} with valid content.
Results stored temporarily at {temp_path} for further processing.
Sample page titles:
{', '.join([r['metadata'].get('title', 'No title') for r in filtered_results[:5]])}
"""

def create_dataset_tool(input_data: Union[str, Dict]) -> str:
    """Tool to create a HuggingFace dataset from crawled content."""
    if isinstance(input_data, str):
        # Parse as JSON if it's a string
        try:
            data = json.loads(input_data)
        except json.JSONDecodeError:
            # If not valid JSON, handle as error
            return f"Invalid JSON input: {input_data}"
    else:
        data = input_data
    
    dataset_name = data.get("dataset_name", "web_crawled_dataset")
    temp_file_path = data.get("temp_file_path")
    push_to_hub = data.get("push_to_hub", False)
    hub_username = data.get("hub_username")
    dataset_description = data.get("dataset_description")
    
    # Load the cached results
    try:
        with open(temp_file_path, 'r') as f:
            dataset_files = json.load(f)
    except Exception as e:
        return f"Error loading dataset files from {temp_file_path}: {str(e)}"
    
    try:
        # Create the dataset
        dataset = create_huggingface_dataset(
            dataset_name=dataset_name,
            dataset_files=dataset_files,
            push_to_hub=push_to_hub,
            hub_username=hub_username,
            dataset_description=dataset_description
        )
        
        return f"""
Successfully created dataset '{dataset_name}' with {len(dataset_files)} documents.
Dataset contains {len(dataset['train'])} entries.
Columns: {list(dataset['train'].features)}
"""
    except Exception as e:
        return f"Error creating dataset: {str(e)}"

def verify_dataset_tool(input_data: Union[str, Dict]) -> str:
    """Tool to verify a dataset exists and check its properties."""
    if isinstance(input_data, str):
        # Parse as JSON if it's a string
        try:
            data = json.loads(input_data)
        except json.JSONDecodeError:
            # If not valid JSON, assume it's just the dataset name
            data = {"dataset_name": input_data}
    else:
        data = input_data
    
    dataset_name = data["dataset_name"]
    
    try:
        # Try to load the dataset
        if "/" in dataset_name:  # Looks like a hub dataset
            dataset = datasets.load_dataset(dataset_name)
        else:
            # Try to load from local path
            dataset = datasets.load_from_disk(dataset_name)
        
        # Get info
        info = {
            "name": dataset_name,
            "splits": list(dataset.keys()),
            "num_rows": {split: len(dataset[split]) for split in dataset},
            "features": str(dataset[list(dataset.keys())[0]].features),
        }
        
        return f"""
Dataset verification successful:
- Name: {info['name']}
- Splits: {', '.join(info['splits'])}
- Number of rows: {info['num_rows']}
- Features: {info['features']}
"""
    except Exception as e:
        return f"Error verifying dataset '{dataset_name}': {str(e)}"

# Create tools for the agent
tools = [
    Tool(
        name="crawl_url",
        func=lambda x: asyncio.run(crawl_url_tool(x)),
        description="Crawl a URL and extract content. Input should be a JSON object with 'url', optional 'max_depth', 'max_pages', 'patterns_to_match' (list of regex), 'patterns_to_exclude' (list of regex)"
    ),
    Tool(
        name="create_dataset",
        func=create_dataset_tool,
        description="Create a HuggingFace dataset from crawled content. Input should be a JSON object with 'dataset_name', 'temp_file_path' (from crawl_url), optional 'push_to_hub', 'hub_username', 'dataset_description'"
    ),
    Tool(
        name="verify_dataset",
        func=verify_dataset_tool,
        description="Verify a dataset exists and check its properties. Input should be a dataset name or path."
    )
]

# Create the LLM using AWS Bedrock
llm = ChatBedrockConverse(
    model_id=MODEL_ID,
    temperature=0.2,
    max_tokens=2000,
)

# Create the agent using LangGraph
def build_agent():
    """Build the dataset creation agent."""
    # Custom system prompt
    system_prompt = """
    You are a specialized Dataset Creator Agent for generating HuggingFace datasets from web content. 
    You help users create high-quality datasets by crawling websites, extracting content, converting HTML to markdown, 
    and generating datasets in the HuggingFace format.

    Your capabilities include:
    1. Crawling websites with customizable depth and URL patterns
    2. Converting HTML content to clean markdown using advanced processing
    3. Creating structured datasets with appropriate metadata
    4. Uploading datasets to the HuggingFace Hub

    You have access to the following tools:
    - crawl_url: Crawl web pages starting from a URL, with configurable depth, max pages, and URL patterns
    - create_dataset: Generate a HuggingFace dataset from crawled content
    - verify_dataset: Verify a dataset exists and check its properties

    IMPORTANT WORKFLOW GUIDELINES:
    1. Always understand the user's dataset creation needs first
    2. Suggest appropriate crawling parameters based on the website structure
    3. Execute the crawl operation with appropriate depth and filtering
    4. Create the dataset with appropriate name and metadata
    5. Verify the dataset was created successfully
    6. Provide clear, step-by-step explanations of what you're doing

    When crawling websites:
    - Respect robots.txt and crawl ethically
    - Set appropriate depth and page limits
    - Use patterns to filter URLs when appropriate
    - Be patient during crawling of larger sites

    When creating datasets:
    - Choose descriptive, clear dataset names
    - Include comprehensive metadata
    - Verify the dataset structure after creation
    """
    
    # Create the agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    
    return agent

def main():
    """Run the agent."""
    agent = build_agent()
    
    # Example invocation
    print("Dataset Creator Agent initialized. You can now ask it to create datasets from URLs.")
    print("Example: Create a dataset from https://www.gov.uk/government/collections/hmrc-manuals")
    
    while True:
        user_input = input("\nYour request (or 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        messages = [("user", user_input)]
        result = agent.invoke({"messages": messages})
        
        # Print the result
        for message in result["messages"]:
            if message[0] == "ai":
                print(f"\nAgent: {message[1]}")

if __name__ == "__main__":
    main()