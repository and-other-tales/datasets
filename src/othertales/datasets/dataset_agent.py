"""
HuggingFace Dataset Creator Agent using LangGraph

This agent creates HuggingFace datasets from URLs by crawling, downloading, and processing content.
It uses LangGraph with PostgreSQL persistence for maintaining state across interactions.

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
from typing import List, Dict, Any, Optional, Union, Literal, TypedDict
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
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

# PostgreSQL persistence
try:
    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("PostgreSQL dependencies not found. Running without persistence.")

# Multi-LLM provider support
from othertales.datasets.llm_utils import get_llm

# For dataset creation
from datasets import Dataset, DatasetDict, Features, Value, Sequence
import datasets

# Configuration 
DEFAULT_MODEL_ID = "anthropic.claude-3-7-sonnet-20250219-v1:0"  # Default for Bedrock
MAX_DEPTH = 3  # Maximum crawling depth
MAX_PAGES = 100  # Maximum number of pages to crawl per domain
TIMEOUT = 30000  # Timeout for page loading in ms

# Set up directories
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dataset_crawler")
os.makedirs(CACHE_DIR, exist_ok=True)

# Agent configuration schema
class AgentConfig(BaseModel):
    """Configuration for the Dataset Creator Agent."""
    system_prompt: str = Field(
        default="You are a specialized Dataset Creator Agent for generating HuggingFace datasets from web content.",
        description="The system prompt to use for the agent's interactions.",
        json_schema_extra={
            "langgraph_nodes": ["call_model"],
            "langgraph_type": "prompt",
        },
    )
    max_depth: int = Field(
        default=MAX_DEPTH,
        description="Maximum crawl depth",
        json_schema_extra={"langgraph_nodes": ["crawl_url"]}
    )
    max_pages: int = Field(
        default=MAX_PAGES,
        description="Maximum pages to crawl per domain",
        json_schema_extra={"langgraph_nodes": ["crawl_url"]}
    )
    patterns_to_match: Optional[List[str]] = Field(
        default=None,
        description="List of regex patterns to include in crawl",
        json_schema_extra={"langgraph_nodes": ["crawl_url"]}
    )
    patterns_to_exclude: Optional[List[str]] = Field(
        default=None,
        description="List of regex patterns to exclude from crawl",
        json_schema_extra={"langgraph_nodes": ["crawl_url"]}
    )

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

# State management for the LangGraph agent
class AgentState(TypedDict):
    """State for the Dataset Creator Agent."""
    messages: List
    crawled_urls: Optional[List[Dict[str, Any]]]
    dataset_info: Optional[Dict[str, Any]]
    temp_file_path: Optional[str]

def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    """Replace SVG content with placeholder text."""
    SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )

def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    """Replace base64 encoded images with a simple image tag."""
    BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)

def clean_html(html_content: str, clean_svg: bool = False, clean_base64: bool = False) -> str:
    """Clean HTML content by removing unnecessary elements and optionally SVG/base64 images."""
    # Patterns
    SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
    STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
    META_PATTERN = r"<[ ]*meta.*?>"
    COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
    LINK_PATTERN = r"<[ ]*link.*?>"
    
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
    
    if clean_svg:
        html_content = replace_svg(html_content)
    if clean_base64:
        html_content = replace_base64_images(html_content)
    
    return html_content

def create_readerlm_prompt(text: str, tokenizer=None) -> str:
    """Create a prompt for the ReaderLM model."""
    instruction = "Extract the main content from the given HTML and convert it to Markdown format."
    prompt = f"{instruction}\n```html\n{text}\n```"
    
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def html_to_markdown(html_content: str) -> str:
    """
    Convert HTML content to markdown using ReaderLM pattern.
    Uses the jinaai/ReaderLM-v2 model for high-quality HTML to markdown conversion.
    Falls back to BeautifulSoup-based conversion if transformers is not available.
    """
    # Clean the HTML first
    html_content = clean_html(html_content, clean_svg=True, clean_base64=True)
    
    try:
        # Try to use the ReaderLM-v2 model from HuggingFace
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Initialize model and tokenizer (with caching)
        model_name = "jinaai/ReaderLM-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # Create input prompt and encode
        input_prompt = create_readerlm_prompt(html_content, tokenizer)
        inputs = tokenizer.encode(input_prompt, return_tensors="pt").to(device)
        
        # Generate markdown with recommended parameters
        outputs = model.generate(
            inputs, 
            max_new_tokens=4096,
            temperature=0,
            do_sample=False,
            repetition_penalty=1.08
        )
        
        # Decode and clean up the output
        markdown_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the markdown part (remove the prompt/instruction)
        if "```" in markdown_text:
            markdown_text = markdown_text.split("```")[-1].strip()
        
        return markdown_text
    
    except (ImportError, Exception) as e:
        print(f"ReaderLM-v2 model error: {str(e)}. Using fallback conversion method.")
        # Fallback to existing BeautifulSoup implementation
        
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

async def recursive_crawl(
    start_url: str, 
    max_depth: int = MAX_DEPTH, 
    max_pages: int = MAX_PAGES, 
    patterns_to_match: List[str] = None,
    patterns_to_exclude: List[str] = None
) -> List[Dict[str, Any]]:
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

# LangGraph node functions
def crawl_url_node(state: AgentState, config: Dict[str, Any]) -> AgentState:
    """Node for crawling URLs and extracting content."""
    # Get the most recent message
    last_message = state["messages"][-1]
    
    # Parse input from the message
    if not isinstance(last_message[1], str):
        # This is a bit complex because we need to handle different message formats
        content = last_message[1].content if hasattr(last_message[1], "content") else str(last_message[1])
    else:
        content = last_message[1]
    
    # Try to parse as JSON
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            data = {"url": content}
    except (json.JSONDecodeError, TypeError):
        # If not valid JSON, assume it's just the URL
        data = {"url": content}
    
    # Get URL and parameters from the data
    url = data.get("url", "")
    if not url:
        # Try to extract URL from text
        import re
        url_match = re.search(r'https?://[^\s]+', content)
        if url_match:
            url = url_match.group(0)
        else:
            # No URL found
            return {
                **state,
                "messages": state["messages"] + [("ai", "Please provide a valid URL to crawl.")]
            }
    
    # Get parameters, either from data or from config
    configurable = config.get("configurable", {})
    max_depth = data.get("max_depth", configurable.get("max_depth", MAX_DEPTH))
    max_pages = data.get("max_pages", configurable.get("max_pages", MAX_PAGES))
    patterns_to_match = data.get("patterns_to_match", configurable.get("patterns_to_match"))
    patterns_to_exclude = data.get("patterns_to_exclude", configurable.get("patterns_to_exclude"))
    
    # Run the crawl
    results = asyncio.run(recursive_crawl(
        url, 
        max_depth=max_depth, 
        max_pages=max_pages,
        patterns_to_match=patterns_to_match,
        patterns_to_exclude=patterns_to_exclude
    ))
    
    # Filter results
    filtered_results = filter_results(results)
    
    # Cache the results for dataset creation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(filtered_results, f)
        temp_path = f.name
    
    # Create summary message
    summary = f"""
Crawled {url} with depth {max_depth}. 
Found {len(results)} pages, {len(filtered_results)} with valid content.
Results stored temporarily for dataset creation.

Sample page titles:
{', '.join([r['metadata'].get('title', 'No title') for r in filtered_results[:5]])}
"""
    
    # Update state
    return {
        **state,
        "crawled_urls": filtered_results,
        "temp_file_path": temp_path,
        "messages": state["messages"] + [("ai", summary)]
    }

def create_dataset_node(state: AgentState, config: Dict[str, Any]) -> AgentState:
    """Node for creating a HuggingFace dataset from crawled content."""
    # Get the most recent message
    last_message = state["messages"][-1]
    
    # Parse input from the message
    if not isinstance(last_message[1], str):
        content = last_message[1].content if hasattr(last_message[1], "content") else str(last_message[1])
    else:
        content = last_message[1]
    
    # Get the crawled URLs from state
    dataset_files = state.get("crawled_urls", [])
    temp_file_path = state.get("temp_file_path")
    
    if not dataset_files and temp_file_path:
        # Load from temp file
        try:
            with open(temp_file_path, 'r') as f:
                dataset_files = json.load(f)
        except Exception as e:
            return {
                **state,
                "messages": state["messages"] + [("ai", f"Error loading dataset files: {str(e)}")]
            }
    
    if not dataset_files:
        return {
            **state,
            "messages": state["messages"] + [("ai", "No crawled URLs available. Please crawl some URLs first.")]
        }
    
    # Try to parse dataset parameters
    try:
        # Try to extract dataset name from the message
        import re
        dataset_name_match = re.search(r'dataset[_\s]?name[:\s]+([a-zA-Z0-9_-]+)', content, re.IGNORECASE)
        if dataset_name_match:
            dataset_name = dataset_name_match.group(1)
        else:
            # Use a default name based on the first URL
            parsed_url = urlparse(dataset_files[0]["url"])
            dataset_name = f"{parsed_url.netloc.replace('.', '_')}_dataset"
        
        # See if we should push to hub
        push_to_hub = "push" in content.lower() and "hub" in content.lower()
        hub_username = None
        
        if push_to_hub:
            # Try to extract hub username
            username_match = re.search(r'username[:\s]+([a-zA-Z0-9_-]+)', content, re.IGNORECASE)
            if username_match:
                hub_username = username_match.group(1)
            else:
                # No username provided
                return {
                    **state,
                    "messages": state["messages"] + [("ai", "To push to HuggingFace Hub, please provide a username.")]
                }
        
        # Extract description if available
        description_match = re.search(r'description[:\s]+(.*?)(?:$|\.|,|\n)', content, re.IGNORECASE)
        dataset_description = description_match.group(1).strip() if description_match else None
        
        # Create the dataset
        dataset = create_huggingface_dataset(
            dataset_name=dataset_name,
            dataset_files=dataset_files,
            push_to_hub=push_to_hub,
            hub_username=hub_username,
            dataset_description=dataset_description
        )
        
        # Create a summary message
        summary = f"""
Successfully created dataset '{dataset_name}' with {len(dataset_files)} documents.
Dataset contains {len(dataset['train'])} entries.
Columns: {list(dataset['train'].features)}

"""
        if push_to_hub:
            summary += f"Dataset pushed to HuggingFace Hub at {hub_username}/{dataset_name}"
        
        # Update state
        return {
            **state,
            "dataset_info": {
                "name": dataset_name,
                "num_entries": len(dataset['train']),
                "columns": list(dataset['train'].features),
                "pushed_to_hub": push_to_hub,
                "hub_path": f"{hub_username}/{dataset_name}" if push_to_hub else None
            },
            "messages": state["messages"] + [("ai", summary)]
        }
        
    except Exception as e:
        return {
            **state,
            "messages": state["messages"] + [("ai", f"Error creating dataset: {str(e)}")]
        }

def verify_dataset_node(state: AgentState, config: Dict[str, Any]) -> AgentState:
    """Node for verifying a dataset exists and checking its properties."""
    # Get the most recent message
    last_message = state["messages"][-1]
    
    # Parse input from the message
    if not isinstance(last_message[1], str):
        content = last_message[1].content if hasattr(last_message[1], "content") else str(last_message[1])
    else:
        content = last_message[1]
    
    # Try to get dataset name
    dataset_name = None
    
    # If we have dataset info in state, use that
    if state.get("dataset_info") and state["dataset_info"].get("name"):
        dataset_name = state["dataset_info"]["name"]
    else:
        # Try to extract dataset name from the message
        import re
        dataset_match = re.search(r'dataset[:\s]+([a-zA-Z0-9_/-]+)', content, re.IGNORECASE)
        if dataset_match:
            dataset_name = dataset_match.group(1)
    
    if not dataset_name:
        return {
            **state,
            "messages": state["messages"] + [("ai", "Please specify a dataset name to verify.")]
        }
    
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
        
        # Create summary message
        summary = f"""
Dataset verification successful:
- Name: {info['name']}
- Splits: {', '.join(info['splits'])}
- Number of rows: {info['num_rows']}
- Features: {info['features']}
"""
        
        # Update state
        return {
            **state,
            "dataset_info": {
                **state.get("dataset_info", {}),
                "verified": True,
                "verification_info": info
            },
            "messages": state["messages"] + [("ai", summary)]
        }
        
    except Exception as e:
        return {
            **state,
            "messages": state["messages"] + [("ai", f"Error verifying dataset '{dataset_name}': {str(e)}")]
        }

# Tool definitions for the agent
def crawl_url_tool(input_data: Union[str, Dict]) -> str:
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
    
    results = asyncio.run(recursive_crawl(
        url, 
        max_depth=max_depth, 
        max_pages=max_pages,
        patterns_to_match=patterns_to_match,
        patterns_to_exclude=patterns_to_exclude
    ))
    
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
        func=crawl_url_tool,
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

# Configure LangSmith tracing for node visibility
import os
from langsmith import traceable
import langsmith
from langchain_core.tracers import ConsoleCallbackHandler

# Set up LangSmith for tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "dataset-creator-agent")

# Create the LLM with tracing
llm = get_llm(callbacks=[ConsoleCallbackHandler()])

# Setup PostgreSQL connection if available
def setup_postgres_connection():
    """Setup PostgreSQL connection for persistence."""
    if not POSTGRES_AVAILABLE:
        return None
    
    # Get database connection details from environment variables
    db_uri = os.environ.get("POSTGRES_URI")
    if not db_uri:
        print("PostgreSQL URI not found in environment variables. Running without persistence.")
        return None
    
    try:
        # Create connection pool
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        
        pool = ConnectionPool(
            conninfo=db_uri,
            max_size=20,
            kwargs=connection_kwargs,
        )
        
        # Create and setup the checkpointer
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()  # Initialize schema
        
        return checkpointer
    except Exception as e:
        print(f"Error setting up PostgreSQL connection: {str(e)}")
        return None

# Create the agent using LangGraph
def build_agent(use_postgres=False, use_tracing=True):
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
    
    # Initialize the checkpointer if PostgreSQL is available
    checkpointer = setup_postgres_connection() if use_postgres else None
    
    # Configure callbacks for LangSmith tracing
    callbacks = []
    runnable_config = {}
    
    if use_tracing:
        from langchain_core.callbacks import CallbackManager
        from langchain_core.tracers import LangChainTracer
        
        tracer = LangChainTracer(project_name=os.environ.get("LANGCHAIN_PROJECT", "dataset-creator-agent"))
        callback_manager = CallbackManager([tracer, ConsoleCallbackHandler()])
        callbacks = [callback_manager]
        
        # Add metadata for visualization
        runnable_config = {
            "tags": ["dataset-agent", "langgraph", "react-agent"],
            "metadata": {
                "agent_type": "dataset-creator",
                "version": "1.0.0",
                "description": "Dataset Creator Agent with LangGraph",
            },
        }
    
    # Create the agent with proper tracing for LangSmith
    # Check LangGraph version and update the create_react_agent arguments as needed
    import inspect
    create_agent_args = {}
    
    # Get the signature of create_react_agent to determine the correct parameters
    create_agent_sig = inspect.signature(create_react_agent)
    if "llm" in create_agent_sig.parameters:
        # Older LangGraph versions
        create_agent_args["llm"] = llm
    else:
        # Newer LangGraph versions use model parameter
        create_agent_args["model"] = llm
        
    # Add common parameters
    create_agent_args.update({
        "tools": tools,
        "system_prompt": system_prompt,
        "checkpointer": checkpointer,
        "callbacks": callbacks,
    })
    
    # Add tool_choice_transition if it's a valid parameter
    if "tool_choice_transition" in create_agent_sig.parameters:
        create_agent_args["tool_choice_transition"] = True
    
    # Create the agent
    agent = create_react_agent(**create_agent_args)
    
    # Add runnable config to enhance visualization
    if use_tracing and hasattr(agent, "with_config"):
        agent = agent.with_config(runnable_config)
    
    return agent

# Apply tracing to node functions
@traceable()
def traced_crawl_url_node(state, config):
    return crawl_url_node(state, config)

@traceable()
def traced_create_dataset_node(state, config):
    return create_dataset_node(state, config)

@traceable()
def traced_verify_dataset_node(state, config):
    return verify_dataset_node(state, config)

# Build the graph explicitly (optional, can use the prebuilt agent instead)
def build_graph(include_tracing=True):
    """Build an explicit LangGraph for the dataset creator agent."""
    # Define the state graph
    builder = StateGraph(AgentState)
    
    # Add nodes with or without tracing
    if include_tracing:
        builder.add_node("crawl_url", traced_crawl_url_node)
        builder.add_node("create_dataset", traced_create_dataset_node)
        builder.add_node("verify_dataset", traced_verify_dataset_node)
        
        # Check if StateGraph.add_node accepts 'llm' or expects 'model'
        import inspect
        add_node_sig = inspect.signature(builder.add_node)
        # This is a simplified check - both methods accept any args,
        # but we're trying to match the pattern of newer LangGraph
        try:
            # For newer LangGraph versions
            builder.add_node("model", llm)
        except Exception:
            # Fallback to older version
            builder.add_node("llm", llm)
    else:
        builder.add_node("crawl_url", crawl_url_node)
        builder.add_node("create_dataset", create_dataset_node)
        builder.add_node("verify_dataset", verify_dataset_node)
        
        # Same logic as above
        try:
            # For newer LangGraph versions
            builder.add_node("model", llm)
        except Exception:
            # Fallback to older version
            builder.add_node("llm", llm)
    
    # Determine if we're using "llm" or "model" as the node name
    llm_node_name = "model"  # Default to newer version
    
    # Check if "model" node exists, otherwise use "llm"
    try:
        if "model" in builder.nodes:
            llm_node_name = "model"
        elif "llm" in builder.nodes:
            llm_node_name = "llm"
    except (AttributeError, TypeError):
        # If we can't check nodes directly, try both in edges
        try:
            # Add edges with "model" node
            builder.add_edge("model", "crawl_url")
            llm_node_name = "model"
        except Exception:
            # Fallback to "llm" node
            builder.add_edge("llm", "crawl_url")
            llm_node_name = "llm"
    
    # Add the remaining edges
    builder.add_edge("crawl_url", "create_dataset")
    builder.add_edge("create_dataset", "verify_dataset")
    builder.add_edge("verify_dataset", llm_node_name)
    
    # Set entry and exit points
    builder.set_entry_point(llm_node_name)
    
    # Compile the graph
    graph = builder.compile()
    
    return graph

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

def create_app():
    """Create FastAPI application with health check endpoint."""
    app = FastAPI()
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/healthz")
    async def healthz():
        """Health check endpoint for Cloud Run."""
        return {"status": "healthy"}
        
    @app.get("/startup")
    async def startup():
        """Startup probe endpoint for Cloud Run."""
        # Check essential dependencies and connections
        health_status = {"status": "healthy", "checks": {}}
        
        # Check PostgreSQL connection if enabled
        if POSTGRES_AVAILABLE and os.environ.get("POSTGRES_URI"):
            try:
                # Attempt to create a connection to test connectivity
                pool = ConnectionPool(
                    conninfo=os.environ.get("POSTGRES_URI"),
                    max_size=1,
                    kwargs={"autocommit": True}
                )
                with pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        conn.commit()
                health_status["checks"]["database"] = "connected"
            except Exception as e:
                health_status["checks"]["database"] = f"error: {str(e)}"
                health_status["status"] = "unhealthy"
        else:
            health_status["checks"]["database"] = "not configured"
        
        # Check LLM connectivity
        try:
            # Simple check to see if LLM provider is configured
            provider = os.environ.get("LLM_PROVIDER", "").lower()
            if provider in ["bedrock", "openai", "anthropic"]:
                health_status["checks"]["llm_provider"] = f"{provider} configured"
            else:
                health_status["checks"]["llm_provider"] = "not configured or unknown"
                health_status["status"] = "unhealthy"
        except Exception as e:
            health_status["checks"]["llm_provider"] = f"error: {str(e)}"
            health_status["status"] = "unhealthy"
        
        # Check LangGraph compatibility
        try:
            # Check if create_react_agent accepts the correct parameters
            import inspect
            from langgraph.prebuilt import create_react_agent
            
            create_agent_sig = inspect.signature(create_react_agent)
            if "model" in create_agent_sig.parameters:
                health_status["checks"]["langgraph_compatibility"] = "compatible (uses model parameter)"
            elif "llm" in create_agent_sig.parameters:
                health_status["checks"]["langgraph_compatibility"] = "compatible (uses llm parameter)"
            else:
                health_status["checks"]["langgraph_compatibility"] = "potentially incompatible"
                health_status["status"] = "unhealthy"
        except Exception as e:
            health_status["checks"]["langgraph_compatibility"] = f"error: {str(e)}"
            health_status["status"] = "unhealthy"
        
        # Return 200 for healthy, 503 for unhealthy
        if health_status["status"] == "healthy":
            return health_status
        else:
            return JSONResponse(content=health_status, status_code=503)
    
    @app.get("/")
    async def root():
        """Root endpoint for Cloud Run health checks."""
        return {"status": "healthy"}
    
    return app

def app(config):
    """LangGraph app factory function that takes a RunnableConfig."""
    fastapi_app = create_app()
    agent = build_agent(use_postgres=False, use_tracing=True)
    
    @fastapi_app.post("/agent")
    async def handle_agent(request: Request):
        """Handle agent requests."""
        body = await request.json()
        messages = body.get("messages", [])
        return agent.invoke({"messages": messages}, config=config)
    
    # Add server configuration
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 2024))
        uvicorn.run(fastapi_app, host="0.0.0.0", port=port)
    
    return fastapi_app