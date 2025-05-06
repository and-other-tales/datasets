"""
Othertales Datasets Agent for HuggingFace

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
from typing import List, Dict, Any, Optional, Union, Literal, TypedDict, Annotated
import hashlib
import tempfile
from pathlib import Path
import time
import datetime 
import uuid
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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

# PostgreSQL persistence
try:
    import psycopg
    from psycopg_pool import ConnectionPool
    try:
        # For newer langgraph versions
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError:
        # For backward compatibility
        from langgraph_checkpoint_postgres import PostgresSaver
    
    POSTGRES_AVAILABLE = True
except ImportError as e:
    POSTGRES_AVAILABLE = False
    print(f"PostgreSQL dependencies not found ({str(e)}). Running without persistence.")
    print("To enable persistence, install: pip install psycopg psycopg_pool langgraph-checkpoint-postgres")

# Multi-LLM provider support
from othertales.datasets.llm_utils import get_llm

# For dataset creation
from datasets import Dataset, DatasetDict, Features, Value, Sequence
import datasets

# FastAPI for web server
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """State for the Dataset Creator Agent."""
    messages: Annotated[List[BaseMessage], add_messages]  # Use Annotated with add_messages reducer
    crawled_urls: Optional[List[Dict[str, Any]]]
    dataset_info: Optional[Dict[str, Any]]
    temp_file_path: Optional[str]
    # Add metadata fields for LangSmith thread tracking
    thread_id: Optional[str]
    session_id: Optional[str]
    metadata: Optional[Dict[str, Any]]

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
from langchain_core.tracers import ConsoleCallbackHandler, LangChainTracer

# Create callback handlers for tracing
callbacks = [ConsoleCallbackHandler()]

# Set up LangSmith for tracing if API key is available
if os.environ.get("LANGSMITH_API_KEY"):
    # Ensure environment variables are set correctly
    os.environ["LANGSMITH_TRACING_V2"] = "true"  # Use V2 tracing protocol
    os.environ["LANGSMITH_ENDPOINT"] = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "datasets")
    
    try:
        # Create and configure a LangChain tracer
        langsmith_tracer = LangChainTracer(
            project_name=os.environ.get("LANGSMITH_PROJECT", "datasets"),
            example_id=None,
        )
        callbacks.append(langsmith_tracer)
        print("LangSmith tracing enabled successfully")
        
        # Register default LangSmith callbacks globally
        from langchain_core.tracers.langsmith import LangSmithCallbackHandler
        from langchain.callbacks.manager import CallbackManager
        
        # This helps ensure we have consistent thread tracking
        global_langsmith_handler = LangSmithCallbackHandler(
            project_name=os.environ.get("LANGSMITH_PROJECT", "datasets"),
        )
        from langchain.globals import set_handler
        set_handler(global_langsmith_handler)
        
        print("Global LangSmith handler registered for thread tracking")
    except Exception as e:
        print(f"Error initializing LangSmith tracing: {str(e)}")
else:
    print("LangSmith API key not found. Tracing disabled.")

# Create the LLM with tracing
llm = get_llm(callbacks=callbacks)

# Create a metadata-aware LLM wrapper for thread tracking
def get_thread_aware_llm(thread_id=None):
    """Get a thread-aware LLM with proper LangSmith metadata."""
    if not thread_id:
        return llm
        
    # Configure metadata with thread_id for LangSmith
    metadata = {"thread_id": thread_id}
    
    # Create a thread-aware LLM wrapper
    from langchain_core.runnables import RunnableConfig
    config = RunnableConfig(
        tags=["dataset-agent", "thread-enabled"],
        metadata=metadata
    )
    
    # Create thread-aware wrapped LLM
    thread_llm = llm.with_config(config)
    return thread_llm

# Setup PostgreSQL connection if available
def setup_postgres_connection():
    """Setup PostgreSQL connection for persistence."""
    if not POSTGRES_AVAILABLE:
        return None
    
    # Get database connection details from environment variables
    db_uri = os.environ.get("DATABASE_URI", os.environ.get("POSTGRES_URI"))
    if not db_uri:
        print("PostgreSQL URI not found in environment variables. Running without persistence.")
        print("Set DATABASE_URI or POSTGRES_URI environment variable to enable persistence.")
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
        try:
            # Initialize schema (wrap in try-except in case schema already exists)
            checkpointer.setup()  
            print("PostgreSQL connection established successfully.")
        except Exception as setup_err:
            print(f"Warning: Schema setup error (schema might already exist): {str(setup_err)}")
        
        return checkpointer
    except Exception as e:
        print(f"Error setting up PostgreSQL connection: {str(e)}")
        print("Continuing without persistence...")
        return None

# Create the agent using LangGraph
def build_agent(use_postgres=False, use_tracing=True, thread_id=None):
    """Build the dataset creation agent with thread support.
    
    Args:
        use_postgres: Whether to use PostgreSQL for persistence
        use_tracing: Whether to enable tracing
        thread_id: Optional thread ID for LangSmith thread tracking
    """
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
    
    # Create the agent arguments 
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import SystemMessage
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Get appropriate LLM - thread-aware if thread_id is provided
    model = get_thread_aware_llm(thread_id) if thread_id else llm
    
    create_agent_args = {
        "model": model,
        "tools": tools,
        "prompt": prompt,
    }
    
    # Add checkpointer if available
    if checkpointer:
        create_agent_args["checkpointer"] = checkpointer
    
    # Create the agent
    try:
        # Try with newer LangGraph API
        agent = create_react_agent(**create_agent_args)
        
        # Configure tracing if enabled
        if use_tracing and hasattr(agent, "with_config"):
            # Add thread tracking metadata for LangSmith if available
            metadata = {
                "agent_type": "dataset-creator",
                "version": "1.0.0",
                "description": "Dataset Creator Agent with LangGraph",
            }
            
            # Add thread metadata for LangSmith if available
            if thread_id:
                metadata["thread_id"] = thread_id
                metadata["session_id"] = thread_id  # LangSmith supports session_id as well
            
            runnable_config = {
                "tags": ["dataset-agent", "langgraph", "react-agent"],
                "metadata": metadata
            }
            agent = agent.with_config(runnable_config)
            
        print(f"Created agent with newer LangGraph API{' (thread-enabled)' if thread_id else ''}")
        
    except Exception as e:
        print(f"Error creating agent with newer API: {str(e)}")
        print("Falling back to legacy mode...")
        
        # Fall back to older version of LangGraph
        create_agent_args["system_message"] = system_prompt
        create_agent_args.pop("prompt", None)
        
        try:
            agent = create_react_agent(**create_agent_args)
            print(f"Created agent with legacy LangGraph API{' (thread-enabled)' if thread_id else ''}")
        except Exception as e2:
            print(f"Failed to create agent with legacy API as well: {str(e2)}")
            raise
    
    return agent

# Apply tracing to node functions with better serialization handling
@traceable(name="crawl_url_node")
def traced_crawl_url_node(state, config):
    """Traceable wrapper for crawl_url_node with proper serialization."""
    try:
        return crawl_url_node(state, config)
    except Exception as e:
        print(f"Error in crawl_url_node: {str(e)}")
        # Return a safe fallback state on error
        return {
            **state,
            "messages": state.get("messages", []) + [("ai", f"Error crawling URL: {str(e)}")]
        }

@traceable(name="create_dataset_node")
def traced_create_dataset_node(state, config):
    """Traceable wrapper for create_dataset_node with proper serialization."""
    try:
        return create_dataset_node(state, config)
    except Exception as e:
        print(f"Error in create_dataset_node: {str(e)}")
        # Return a safe fallback state on error
        return {
            **state,
            "messages": state.get("messages", []) + [("ai", f"Error creating dataset: {str(e)}")]
        }

@traceable(name="verify_dataset_node") 
def traced_verify_dataset_node(state, config):
    """Traceable wrapper for verify_dataset_node with proper serialization."""
    try:
        return verify_dataset_node(state, config)
    except Exception as e:
        print(f"Error in verify_dataset_node: {str(e)}")
        # Return a safe fallback state on error
        return {
            **state,
            "messages": state.get("messages", []) + [("ai", f"Error verifying dataset: {str(e)}")]
        }

# Build the graph explicitly (optional, can use the prebuilt agent instead)
def build_graph(include_tracing=True, use_postgres=False, thread_id=None):
    """Build an explicit LangGraph for the dataset creator agent with thread support.
    
    Args:
        include_tracing: Whether to enable tracing
        use_postgres: Whether to use PostgreSQL for persistence
        thread_id: Optional thread ID for LangSmith thread tracking
    """
    try:
        # Try with new API (LangGraph 0.4+)
        from langgraph.graph import START, END
        
        # Define the state graph
        builder = StateGraph(AgentState)
        
        # Add nodes with or without tracing
        # Using 'model' as the standard node name for LangGraph 0.4+
        llm_node_name = "model"
        
        # Get thread-aware LLM if thread_id is provided
        model = get_thread_aware_llm(thread_id) if thread_id else llm
        
        if include_tracing:
            builder.add_node("crawl_url", traced_crawl_url_node)
            builder.add_node("create_dataset", traced_create_dataset_node)
            builder.add_node("verify_dataset", traced_verify_dataset_node)
            builder.add_node(llm_node_name, model)
        else:
            builder.add_node("crawl_url", crawl_url_node)
            builder.add_node("create_dataset", create_dataset_node)
            builder.add_node("verify_dataset", verify_dataset_node)
            builder.add_node(llm_node_name, model)
        
        # Add edges with consistent node naming
        builder.add_edge(START, llm_node_name)
        builder.add_edge(llm_node_name, "crawl_url")
        builder.add_edge("crawl_url", "create_dataset")
        builder.add_edge("create_dataset", "verify_dataset")
        builder.add_edge("verify_dataset", llm_node_name)
        
        # Add optional edges (for visualization)
        builder.add_edge("verify_dataset", END, condition=lambda state: state.get("dataset_info", {}).get("verified", False))
        
        # Compile options
        compile_options = {}
        
        # Add checkpointer if postgres is available
        if use_postgres:
            checkpointer = setup_postgres_connection()
            if checkpointer:
                compile_options["checkpointer"] = checkpointer
        
        # Add config with tracing information
        if include_tracing:
            # Prepare metadata with thread information if available
            metadata = {
                "agent_type": "dataset-creator",
                "version": "1.0.0",
                "description": "Dataset Creator Agent with LangGraph"
            }
            
            # Add thread metadata for LangSmith if available
            if thread_id:
                metadata["thread_id"] = thread_id
                metadata["session_id"] = thread_id  # LangSmith supports session_id as well
                
            compile_options["config"] = {
                "tags": ["dataset-agent", "langgraph"],
                "metadata": metadata
            }
        
        # Compile the graph with configuration
        graph = builder.compile(**compile_options)
        
        print(f"Built graph with new LangGraph 0.4+ API{' (thread-enabled)' if thread_id else ''}")
        return graph
        
    except (ImportError, Exception) as e:
        print(f"Error building graph with new API: {str(e)}. Falling back to legacy API...")
        
        try:
            # Fall back to legacy API (LangGraph < 0.4)
            # Define the state graph
            builder = StateGraph(AgentState)
            
            # Add nodes with or without tracing
            llm_node_name = "agent"  # Using 'agent' as the standard node name for older versions
            
            # Get thread-aware LLM if thread_id is provided
            model = get_thread_aware_llm(thread_id) if thread_id else llm
            
            if include_tracing:
                builder.add_node("crawl_url", traced_crawl_url_node)
                builder.add_node("create_dataset", traced_create_dataset_node)
                builder.add_node("verify_dataset", traced_verify_dataset_node)
                builder.add_node(llm_node_name, model)
            else:
                builder.add_node("crawl_url", crawl_url_node)
                builder.add_node("create_dataset", create_dataset_node)
                builder.add_node("verify_dataset", verify_dataset_node)
                builder.add_node(llm_node_name, model)
            
            # Add edges with consistent node naming
            builder.add_edge(llm_node_name, "crawl_url")
            builder.add_edge("crawl_url", "create_dataset")
            builder.add_edge("create_dataset", "verify_dataset")
            builder.add_edge("verify_dataset", llm_node_name)
            
            # Set entry point (legacy method)
            builder.set_entry_point(llm_node_name)
            
            # Compile options
            compile_options = {}
            
            # Add checkpointer if postgres is available
            if use_postgres:
                checkpointer = setup_postgres_connection()
                if checkpointer:
                    compile_options["checkpointer"] = checkpointer
            
            # Add config with tracing information
            if include_tracing:
                # Prepare metadata with thread information if available
                metadata = {
                    "agent_type": "dataset-creator",
                    "version": "1.0.0",
                    "description": "Dataset Creator Agent with LangGraph"
                }
                
                # Add thread metadata for LangSmith if available
                if thread_id:
                    metadata["thread_id"] = thread_id
                    metadata["session_id"] = thread_id  # LangSmith supports session_id as well
                    
                compile_options["config"] = {
                    "tags": ["dataset-agent", "langgraph"],
                    "metadata": metadata
                }
            
            # Compile the graph with configuration
            graph = builder.compile(**compile_options)
            
            print(f"Built graph with legacy LangGraph API{' (thread-enabled)' if thread_id else ''}")
            return graph
        
        except Exception as e2:
            print(f"Error building graph with legacy API as well: {str(e2)}")
            raise

def app(config=None):
    """LangGraph app factory function."""
    # Directly return the FastAPI application
    return create_app()

async def test_thread_tracking(thread_id=None):
    """Test function to verify thread tracking is working correctly.
    
    Args:
        thread_id: Optional thread ID to test with. If not provided, a new one will be created.
    
    Returns:
        dict: A dictionary containing test results
    """
    # Create a thread ID if none provided
    if not thread_id:
        thread_id = str(uuid.uuid4())
        
    print(f"Testing thread tracking with thread_id: {thread_id}")
    
    # Create a thread-aware agent
    thread_agent = build_agent(use_tracing=True, thread_id=thread_id)
    
    # Create a few test inputs to send to the agent
    test_inputs = [
        {"messages": [("user", "What can you help me with?")]},
        {"messages": [("user", "How would I create a dataset from example.com?")]},
        {"messages": [("user", "Can I export that dataset to HuggingFace?")]}
    ]
    
    # Process each input in sequence, maintaining thread context
    results = []
    for i, test_input in enumerate(test_inputs):
        print(f"Processing test input {i+1}/{len(test_inputs)} in thread {thread_id}")
        
        # Add thread context to input
        test_input["thread_id"] = thread_id
        test_input["session_id"] = thread_id
        test_input["metadata"] = {"thread_id": thread_id, "session_id": thread_id}
        
        # Create config with thread info
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {"thread_id": thread_id, "session_id": thread_id}
        }
        
        # Process the input
        try:
            response = await thread_agent.ainvoke(test_input, config)
            
            # Extract result text
            if hasattr(response, "content"):
                result = response.content
            elif isinstance(response, dict) and "messages" in response:
                messages = response.get("messages", [])
                if messages and len(messages) > 0:
                    last_message = messages[-1]
                    if isinstance(last_message, tuple) and len(last_message) >= 2:
                        result = str(last_message[1])
                    elif hasattr(last_message, "content"):
                        result = last_message.content
                    else:
                        result = str(messages)
                else:
                    result = str(response)
            else:
                result = str(response)
                
            results.append({
                "input": test_input["messages"][-1][1],
                "response": result,
                "success": True
            })
        except Exception as e:
            results.append({
                "input": test_input["messages"][-1][1],
                "error": str(e),
                "success": False
            })
            
    print(f"Completed thread tracking test for thread: {thread_id}")
    print(f"Results: {len([r for r in results if r['success']])} successful, {len([r for r in results if not r['success']])} failed")
    
    return {
        "thread_id": thread_id,
        "results": results,
        "success_rate": len([r for r in results if r["success"]]) / len(results if results else 0),
        "timestamp": datetime.datetime.now().isoformat()
    }

def create_app():
    """Create FastAPI application."""
    app = FastAPI(
        title="OtherTales Dataset Creator",
        description="Creates HuggingFace datasets from web content using LangGraph",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Fixed assistant ID for the dataset creator
    ASSISTANT_ID = "435f210f-5fb2-51f2-bac9-1b55c4915b36"
    
    @app.get("/healthz")
    async def healthz():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"status": "healthy"}
    
    @app.get("/startup")
    async def startup():
        """Startup probe endpoint."""
        return {"status": "ready"}
        
    @app.post("/test-thread-tracking")
    async def test_langsmith_threads(request: Request):
        """Test endpoint for verifying LangSmith thread tracking."""
        try:
            body = await request.json()
            thread_id = body.get("thread_id")
            
            # Run the test
            results = await test_thread_tracking(thread_id)
            
            return JSONResponse(content=results)
        except Exception as e:
            import traceback
            error_msg = f"Error in thread tracking test: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
        
    @app.post("/assistants/search")
    async def search_assistants(request: Request):
        """Search assistants endpoint."""
        try:
            # Parse request according to schema
            query_params = await request.json()
            
            limit = query_params.get("limit", 10)
            offset = query_params.get("offset", 0)
            metadata_filter = query_params.get("metadata", {})
            graph_id_filter = query_params.get("graph_id", None)
            
            # Create a fixed response with proper format matching schema
            assistant_id = "435f210f-5fb2-51f2-bac9-1b55c4915b36"  # Using a fixed UUID format
            
            # Return a fixed result for now - just one assistant
            assistants = [{
                "assistant_id": assistant_id,
                "graph_id": "dataset_creator",
                "name": "Dataset Creator Agent",
                "config": {
                    "tags": ["dataset", "crawl", "huggingface"],
                    "configurable": {
                        "max_depth": MAX_DEPTH,
                        "max_pages": MAX_PAGES,
                        "patterns_to_match": None,
                        "patterns_to_exclude": None
                    }
                },
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "metadata": {
                    "description": "Creates HuggingFace datasets from web content",
                    "capabilities": [
                        "crawl_url", 
                        "create_dataset",
                        "verify_dataset"
                    ],
                    "model": os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-7-sonnet-20250219-v1:0")
                },
                "version": 1
            }]
            
            # Apply pagination
            paginated_assistants = assistants[offset:offset+limit]
            
            # Return directly as array according to schema
            return JSONResponse(content=paginated_assistants)
        
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )

    # Agent cache to support thread-based continuations
    agent_cache = {}
    
    # Helper function to get or create an agent with proper thread ID
    def get_or_create_agent(thread_id=None):
        """Get or create an agent for the given thread ID."""
        if not thread_id:
            # For stateless requests, use the default agent
            return build_agent(use_postgres=False, use_tracing=True)
            
        # For thread-based requests, check cache or create new agent
        if thread_id not in agent_cache:
            agent_cache[thread_id] = build_agent(use_postgres=False, use_tracing=True, thread_id=thread_id)
            
        return agent_cache[thread_id]
    
    # Default agent for backward compatibility
    default_agent = build_agent(use_postgres=False, use_tracing=True)
    
    @app.post("/assistants/{assistant_id}")
    async def handle_assistant_post(assistant_id: str, request: Request):
        """Handle assistant POST requests."""
        try:
            body = await request.json()
            
            # We're going to continue handling the LangChain invocations,
            # but make sure we handle the format correctly
            
            # Check if body contains pagination parameters but no valid message content
            if ("limit" in body or "offset" in body) and "messages" not in body:
                # Extract pagination parameters
                limit = body.get("limit", 1000)
                offset = body.get("offset", 0)
                
                # Return empty result with pagination info
                return JSONResponse(content={
                    "content": "No messages to process",
                    "type": "text",
                    "pagination": {"limit": limit, "offset": offset}
                })
            
            # Get the thread ID for LangSmith tracking
            thread_id = body.get("thread_id", str(uuid.uuid4()))
            
            # Ensure proper format for input - handle both string inputs and structured messages
            input_data = {}
            
            # Check for direct string input - preferred format for schema compatibility
            if isinstance(body, str):
                # Direct string input - simplest case
                input_data["messages"] = [("user", body)]
            elif "messages" in body:
                # Fix message format for LangGraph
                messages_list = []
                if isinstance(body["messages"], list):
                    for msg in body["messages"]:
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            # Convert to tuple format for compatibility
                            messages_list.append((msg["role"], msg["content"]))
                        elif isinstance(msg, tuple) and len(msg) == 2:
                            messages_list.append(msg)
                        elif isinstance(msg, str):
                            # Default to user role for string messages
                            messages_list.append(("user", msg))
                elif isinstance(body["messages"], dict) and "role" in body["messages"] and "content" in body["messages"]:
                    # Single message as dict
                    messages_list.append((body["messages"]["role"], body["messages"]["content"]))
                elif isinstance(body["messages"], str):
                    # Single message as string
                    messages_list.append(("user", body["messages"]))
                
                input_data["messages"] = messages_list
            elif "input" in body and isinstance(body["input"], str):
                # Handle direct input field as string
                input_data["messages"] = [("user", body["input"])]
            else:
                # If no messages key, wrap the content as a user message
                user_content = json.dumps(body) if isinstance(body, dict) else str(body)
                input_data["messages"] = [("user", user_content)]
            
            # Add thread_id and session_id to input state for LangSmith tracking
            input_data["thread_id"] = thread_id
            input_data["session_id"] = thread_id
            input_data["metadata"] = {"thread_id": thread_id, "session_id": thread_id}
            
            # Prepare config with enhanced metadata for LangSmith
            config = {
                "configurable": {
                    "thread_id": thread_id
                },
                "metadata": {
                    "thread_id": thread_id,
                    "session_id": thread_id
                }
            }
            
            # Get or create an agent specific to this thread for continuity
            agent = get_or_create_agent(thread_id)
            
            # Invoke the agent with properly formatted input
            response = await agent.ainvoke(input_data, config)
            
            # Better handling of different response types for LangSmith compatibility
            if hasattr(response, "content"):
                serialized_response = {
                    "content": response.content,
                    "type": "ai_message",
                    "additional_kwargs": getattr(response, "additional_kwargs", {}),
                    "thread_id": thread_id
                }
            elif isinstance(response, dict) and "messages" in response:
                # Handle LangGraph state dictionary with messages
                messages = response.get("messages", [])
                if messages and len(messages) > 0:
                    last_message = messages[-1]
                    if isinstance(last_message, tuple) and len(last_message) >= 2:
                        serialized_response = {
                            "content": str(last_message[1]),
                            "type": "message",
                            "role": str(last_message[0]),
                            "thread_id": thread_id
                        }
                    elif hasattr(last_message, "content"):
                        # Handle BaseMessage objects
                        serialized_response = {
                            "content": last_message.content,
                            "type": "message",
                            "role": getattr(last_message, "type", "assistant"),
                            "additional_kwargs": getattr(last_message, "additional_kwargs", {}),
                            "thread_id": thread_id
                        }
                    else:
                        serialized_response = {
                            "content": str(messages),
                            "type": "messages",
                            "thread_id": thread_id
                        }
                else:
                    serialized_response = {
                        "content": str(response),
                        "type": "state",
                        "thread_id": thread_id
                    }
            elif isinstance(response, (list, tuple)):
                # Ensure all elements are properly stringified for JSON
                serialized_response = {
                    "content": [str(r) for r in response],
                    "type": "list",
                    "thread_id": thread_id
                }
            else:
                # Default case - convert to string to ensure serialization works
                serialized_response = {
                    "content": str(response),
                    "type": "text",
                    "thread_id": thread_id
                }
            
            return JSONResponse(content=serialized_response)
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing request: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
    
    @app.get("/assistants/{assistant_id}")
    async def handle_assistant_get(assistant_id: str, request: Request):
        """Handle assistant GET requests - returns assistant by ID."""
        try:
            # Check if assistant_id matches our fixed ID
            if assistant_id != ASSISTANT_ID:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Assistant with ID {assistant_id} not found"}
                )
                
            # Create an assistant response matching the OpenAPI schema
            # Using same format as search endpoint
            serialized_response = {
                "assistant_id": assistant_id,
                "graph_id": "dataset_creator",
                "name": "Dataset Creator Agent", 
                "config": {
                    "tags": ["dataset", "crawl", "huggingface"],
                    "configurable": {
                        "max_depth": MAX_DEPTH,
                        "max_pages": MAX_PAGES,
                        "patterns_to_match": None,
                        "patterns_to_exclude": None
                    }
                },
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "metadata": {
                    "description": "Creates HuggingFace datasets from web content",
                    "capabilities": [
                        "crawl_url", 
                        "create_dataset",
                        "verify_dataset"
                    ],
                    "model": os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-7-sonnet-20250219-v1:0")
                },
                "version": 1
            }
            
            return JSONResponse(content=serialized_response)
            
        except Exception as e:
            return JSONResponse(
                status_code=404 if "not found" in str(e).lower() else 500,
                content={"error": str(e)}
            )
    
    @app.delete("/assistants/{assistant_id}")
    async def delete_assistant(assistant_id: str):
        """Delete assistant by ID."""
        if assistant_id != ASSISTANT_ID:
            return JSONResponse(
                status_code=404,
                content={"error": f"Assistant with ID {assistant_id} not found"}
            )
        
        # Return an empty response to indicate success
        return JSONResponse(content={})
    
    @app.patch("/assistants/{assistant_id}")
    async def patch_assistant(assistant_id: str, request: Request):
        """Update assistant with provided fields."""
        try:
            if assistant_id != ASSISTANT_ID:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Assistant with ID {assistant_id} not found"}
                )
                
            body = await request.json()
            
            # Start with the current configuration
            assistant_config = {
                "tags": ["dataset", "crawl", "huggingface"],
                "configurable": {
                    "max_depth": MAX_DEPTH,
                    "max_pages": MAX_PAGES,
                    "patterns_to_match": None,
                    "patterns_to_exclude": None
                }
            }
            
            # If config is provided, update the configurable parameters
            if "config" in body and isinstance(body["config"], dict):
                if "configurable" in body["config"] and isinstance(body["config"]["configurable"], dict):
                    for key, value in body["config"]["configurable"].items():
                        if key in assistant_config["configurable"]:
                            assistant_config["configurable"][key] = value
                
                # Update tags if provided
                if "tags" in body["config"] and isinstance(body["config"]["tags"], list):
                    assistant_config["tags"] = body["config"]["tags"]
            
            # Merge metadata with existing metadata
            base_metadata = {
                "description": "Creates HuggingFace datasets from web content",
                "capabilities": [
                    "crawl_url", 
                    "create_dataset",
                    "verify_dataset"
                ],
                "model": os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-7-sonnet-20250219-v1:0")
            }
            
            if "metadata" in body and isinstance(body["metadata"], dict):
                # Merge the metadata, with new values taking precedence
                for key, value in body["metadata"].items():
                    base_metadata[key] = value
            
            serialized_response = {
                "assistant_id": assistant_id,
                "graph_id": body.get("graph_id", "dataset_creator"),
                "name": body.get("name", "Dataset Creator Agent"), 
                "config": assistant_config,
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "metadata": base_metadata,
                "version": 1
            }
            
            return JSONResponse(content=serialized_response)
        except Exception as e:
            return JSONResponse(
                status_code=422,
                content={"error": str(e)}
            )
    
    @app.get("/assistants/{assistant_id}/graph")
    async def get_assistant_graph(assistant_id: str, xray: bool = False):
        """Get assistant graph."""
        if assistant_id != ASSISTANT_ID and assistant_id != "dataset_creator":
            return JSONResponse(
                status_code=404,
                content={"error": f"Assistant with ID {assistant_id} not found"}
            )
        
        # Return an enhanced graph representation with proper connections for visualization
        # Define message list schema for graph inputs/outputs
        message_item_schema = {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "Role of the message sender (e.g., user, system, assistant)"},
                "content": {"type": "string", "description": "Message content text"}
            },
            "required": ["role", "content"]
        }
        messages_schema = {"type": "array", "items": message_item_schema}
        state_input_schema = {
            "type": "object",
            "properties": {"messages": messages_schema},
            "required": ["messages"],
            "description": "Initial state with a list of messages"
        }
        graph_representation = {
            "nodes": [
                {"id": "__start__", "type": "start", "display_name": "Start", "input_schema": state_input_schema, "output_schema": state_input_schema},
                {"id": "model", "type": "llm", "display_name": "LLM", "input_schema": {"type": "string"}, "output_schema": {"type": "string"}},
                {"id": "crawl_url", "type": "tool", "display_name": "Crawl URL", "input_schema": {"type": "string"}, "output_schema": {"type": "string"}},
                {"id": "create_dataset", "type": "tool", "display_name": "Create Dataset", "input_schema": {"type": "string"}, "output_schema": {"type": "string"}},
                {"id": "verify_dataset", "type": "tool", "display_name": "Verify Dataset", "input_schema": {"type": "string"}, "output_schema": {"type": "string"}},
                {"id": "__end__", "type": "end", "display_name": "End", "input_schema": {"type": "string"}, "output_schema": {"type": "string"}}
            ],
            "edges": [
                {"from": "__start__", "to": "model", "label": "start"},
                {"from": "model", "to": "crawl_url", "label": "crawl"},
                {"from": "crawl_url", "to": "create_dataset", "label": "create"},
                {"from": "create_dataset", "to": "verify_dataset", "label": "verify"},
                {"from": "verify_dataset", "to": "model", "label": "continue", "condition": "not verified"},
                {"from": "verify_dataset", "to": "__end__", "label": "end", "condition": "verified"}
            ],
            "node_types": {
                "start": {"color": "#e0f7fa", "shape": "circle"},
                "llm": {"color": "#bbdefb", "shape": "rectangle"},
                "tool": {"color": "#c8e6c9", "shape": "rectangle"},
                "end": {"color": "#ffccbc", "shape": "circle"}
            },
            "layout": "directed",
            "version": "1.0.0",
            "graph_schema": {
                # Graph expects initial state with messages key
                "input_schema": state_input_schema,
                # Graph output returns updated state, include messages key
                "output_schema": state_input_schema
            }
        }
        
        return JSONResponse(content=graph_representation)
    
    @app.get("/assistants/{assistant_id}/schemas")
    async def get_assistant_schemas(assistant_id: str):
        """Get assistant schemas."""
        if assistant_id != ASSISTANT_ID:
            return JSONResponse(
                status_code=404,
                content={"error": f"Assistant with ID {assistant_id} not found"}
            )
        
        # Return a schema representation with updated format for chat support
        # Define schema for incoming messages array
        message_item_schema = {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "Role of the message sender (e.g., user, system, assistant)"},
                "content": {"type": "string", "description": "Message content text"}
            },
            "required": ["role", "content"]
        }
        messages_schema = {"type": "array", "items": message_item_schema}
        schema = {
            "graph_id": "dataset_creator",
            "input_schema": {
                "type": "object",
                "properties": {"messages": messages_schema},
                "required": ["messages"],
                "description": "Input must be an object with a 'messages' key containing an array of message objects"
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "type": {"type": "string"},
                    "role": {"type": "string"},
                    "additional_kwargs": {"type": "object"}
                }
            },
            "state_schema": {
                "type": "object",
                "properties": {
                    "messages": {"type": "array"},
                    "crawled_urls": {"type": "array"},
                    "dataset_info": {"type": "object"},
                    "temp_file_path": {"type": "string"},
                    "thread_id": {"type": "string"},
                    "session_id": {"type": "string"},
                    "metadata": {"type": "object"}
                }
            },
            "node_schemas": {
                "__start__": {
                    "input_schema": {"type": "string"},
                    "output_schema": {"type": "string"}
                },
                "model": {
                    "input_schema": {"type": "string"},
                    "output_schema": {"type": "string"}
                },
                "crawl_url": {
                    "input_schema": {"type": "string"},
                    "output_schema": {"type": "string"}
                },
                "create_dataset": {
                    "input_schema": {"type": "string"},
                    "output_schema": {"type": "string"}
                },
                "verify_dataset": {
                    "input_schema": {"type": "string"},
                    "output_schema": {"type": "string"}
                },
                "__end__": {
                    "input_schema": {"type": "string"},
                    "output_schema": {"type": "string"}
                }
            },
            "config_schema": {
                "type": "object",
                "properties": {
                    "configurable": {
                        "type": "object",
                        "properties": {
                            "thread_id": {"type": "string"},
                            "user_id": {"type": "string"},
                            "max_depth": {"type": "integer"},
                            "max_pages": {"type": "integer"},
                            "patterns_to_match": {"type": "array"},
                            "patterns_to_exclude": {"type": "array"}
                        }
                    }
                }
            }
        }
        
        return JSONResponse(content=schema)
        
    @app.post("/threads")
    async def create_thread(request: Request):
        """Create a new thread for storing conversation state."""
        try:
            body = await request.json()
            
            # Generate a thread ID if not provided
            thread_id = body.get("thread_id", str(uuid.uuid4()))
            
            # Store thread creation time
            created_at = datetime.datetime.now().isoformat()
            
            # Extract metadata or use empty dict
            metadata = body.get("metadata", {})
            
            # Ensure the metadata has the necessary fields for LangSmith thread tracking
            if "thread_id" not in metadata:
                metadata["thread_id"] = thread_id
            if "session_id" not in metadata:
                metadata["session_id"] = thread_id
                
            # Create thread response object
            thread = {
                "thread_id": thread_id,
                "created_at": created_at,
                "updated_at": created_at,
                "metadata": metadata,
                "status": "idle",
                "values": {}
            }
            
            # Create a thread-specific agent for this thread_id
            if thread_id not in agent_cache:
                agent_cache[thread_id] = build_agent(use_postgres=False, use_tracing=True, thread_id=thread_id)
                print(f"Created new thread-specific agent for thread: {thread_id}")
            
            # Register the thread with LangSmith if LangSmith tracing is enabled
            if os.environ.get("LANGSMITH_API_KEY"):
                try:
                    import langsmith
                    from langsmith.client import Client
                    
                    client = Client()
                    
                    # Create a run in this thread to register it with LangSmith
                    langsmith_metadata = {
                        "thread_id": thread_id,
                        "session_id": thread_id,
                        "chat_initialized": True,
                        "agent_type": "dataset-creator"
                    }
                    
                    # Update our thread object with LangSmith-specific fields
                    thread["langsmith_metadata"] = langsmith_metadata
                    
                    print(f"Registered thread {thread_id} with LangSmith")
                except Exception as e:
                    print(f"Error registering thread with LangSmith: {str(e)}")
            
            return JSONResponse(content=thread)
        except Exception as e:
            return JSONResponse(
                status_code=422,
                content={"error": str(e)}
            )
        
    @app.post("/threads/search")
    async def search_threads(request: Request):
        """Search threads based on criteria."""
        try:
            body = await request.json()
            
            # Extract search parameters
            limit = body.get("limit", 10)
            offset = body.get("offset", 0)
            metadata_filter = body.get("metadata", {})
            values_filter = body.get("values", {})
            status_filter = body.get("status")
            
            # Since this is a stateless implementation, we return an empty array
            # A database-backed implementation would filter threads here
            return JSONResponse(content=[])
        except Exception as e:
            return JSONResponse(
                status_code=422,
                content={"error": str(e)}
            )
    
    @app.get("/threads/{thread_id}")
    async def get_thread(thread_id: str):
        """Get thread by ID."""
        try:
            # In this simple implementation, we construct a thread object
            # A database-backed implementation would retrieve the thread data
            thread = {
                "thread_id": thread_id,
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "metadata": {},
                "status": "idle",
                "values": {}
            }
            
            return JSONResponse(content=thread)
        except Exception as e:
            return JSONResponse(
                status_code=404,
                content={"error": f"Thread {thread_id} not found"}
            )
    
    @app.post("/threads/{thread_id}/runs")
    async def create_run(thread_id: str, request: Request):
        """Create a run in an existing thread."""
        try:
            body = await request.json()
            
            # Validate the assistant ID
            assistant_id = body.get("assistant_id")
            if assistant_id != ASSISTANT_ID and assistant_id != "dataset_creator":
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Assistant with ID {assistant_id} not found"}
                )
            
            # Generate a run ID
            run_id = str(uuid.uuid4())
            created_at = datetime.datetime.now().isoformat()
            
            # Extract metadata from body or create empty dict
            metadata = body.get("metadata", {})
            
            # Ensure the metadata has the necessary fields for LangSmith thread tracking
            if "thread_id" not in metadata:
                metadata["thread_id"] = thread_id
            if "session_id" not in metadata:
                metadata["session_id"] = thread_id
            
            # Create or get thread-specific agent for this thread_id
            if thread_id not in agent_cache:
                agent_cache[thread_id] = build_agent(use_postgres=False, use_tracing=True, thread_id=thread_id)
                print(f"Created new thread-specific agent for thread: {thread_id}")
                
            # Register the run with LangSmith if LangSmith tracing is enabled
            langsmith_run_id = None
            if os.environ.get("LANGSMITH_API_KEY"):
                try:
                    import langsmith
                    from langsmith.client import Client
                    
                    client = Client()
                    
                    # Add LangSmith tracking metadata
                    langsmith_metadata = {
                        "thread_id": thread_id,
                        "session_id": thread_id,
                        "run_id": run_id,
                        "assistant_id": assistant_id,
                        "agent_type": "dataset-creator"
                    }
                    
                    # Add LangSmith run info to metadata
                    metadata.update(langsmith_metadata)
                    
                    print(f"Added LangSmith tracking for run {run_id} in thread {thread_id}")
                except Exception as e:
                    print(f"Error adding LangSmith tracking: {str(e)}")
            
            # Create run response
            run = {
                "run_id": run_id,
                "thread_id": thread_id,
                "assistant_id": assistant_id,
                "created_at": created_at,
                "updated_at": created_at,
                "status": "pending",
                "metadata": metadata,
                "kwargs": {},
                "multitask_strategy": body.get("multitask_strategy", "reject")
            }
            
            # If LangSmith run was created, add it to the response
            if langsmith_run_id:
                run["langsmith_run_id"] = langsmith_run_id
            
            return JSONResponse(content=run)
        except Exception as e:
            return JSONResponse(
                status_code=422,
                content={"error": str(e)}
            )
    
    @app.post("/threads/{thread_id}/runs/wait")
    async def wait_run(thread_id: str, request: Request):
        """Create a run and wait for completion."""
        try:
            body = await request.json()
            
            # Validate the assistant ID
            assistant_id = body.get("assistant_id")
            if assistant_id != ASSISTANT_ID and assistant_id != "dataset_creator":
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Assistant with ID {assistant_id} not found"}
                )
            
            # Process the input using our agent
            input_data = body.get("input", {})
            
            # Format input for the agent - prioritize string input for schema compatibility
            if isinstance(input_data, str):
                # Direct string input - preferred format for schema compatibility
                formatted_input = {"messages": [("user", input_data)]}
            elif isinstance(input_data, dict) and "messages" in input_data:
                # Process structured messages
                if isinstance(input_data["messages"], list):
                    messages_list = []
                    for msg in input_data["messages"]:
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            messages_list.append((msg["role"], msg["content"]))
                        elif isinstance(msg, str):
                            messages_list.append(("user", msg))
                    formatted_input = {"messages": messages_list}
                elif isinstance(input_data["messages"], str):
                    formatted_input = {"messages": [("user", input_data["messages"])]}
                else:
                    formatted_input = input_data
            else:
                # Default case - convert to string
                user_content = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data)
                formatted_input = {"messages": [("user", user_content)]}
            
            # Process through agent
            response = await agent.ainvoke(formatted_input)
            
            # Process the response
            if hasattr(response, "content"):
                result = response.content
            elif isinstance(response, dict) and "messages" in response:
                messages = response.get("messages", [])
                if messages and len(messages) > 0:
                    last_message = messages[-1]
                    if isinstance(last_message, tuple) and len(last_message) >= 2:
                        result = str(last_message[1])
                    else:
                        result = str(messages)
                else:
                    result = str(response)
            elif isinstance(response, (list, tuple)):
                result = [str(r) for r in response]
            else:
                result = str(response)
            
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(
                status_code=422,
                content={"error": str(e)}
            )
    
    @app.post("/runs")
    async def create_stateless_run(request: Request):
        """Create a stateless run (no persistent thread)."""
        try:
            body = await request.json()
            
            # Validate the assistant ID
            assistant_id = body.get("assistant_id")
            if assistant_id != ASSISTANT_ID and assistant_id != "dataset_creator":
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Assistant with ID {assistant_id} not found"}
                )
            
            # Generate IDs
            run_id = str(uuid.uuid4())
            thread_id = str(uuid.uuid4())
            created_at = datetime.datetime.now().isoformat()
            
            # Create run response
            run = {
                "run_id": run_id,
                "thread_id": thread_id,
                "assistant_id": assistant_id,
                "created_at": created_at,
                "updated_at": created_at,
                "status": "pending",
                "metadata": body.get("metadata", {}),
                "kwargs": {},
                "multitask_strategy": body.get("multitask_strategy", "reject")
            }
            
            return JSONResponse(content=run)
        except Exception as e:
            return JSONResponse(
                status_code=422,
                content={"error": str(e)}
            )
    
    @app.post("/runs/wait")
    async def wait_stateless_run(request: Request):
        """Create a stateless run and wait for completion."""
        try:
            body = await request.json()
            
            # Validate the assistant ID
            assistant_id = body.get("assistant_id")
            if assistant_id != ASSISTANT_ID and assistant_id != "dataset_creator":
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Assistant with ID {assistant_id} not found"}
                )
            
            # Process the input using our agent
            input_data = body.get("input", {})
            
            # Get thread_id if available for LangSmith tracking
            thread_id = body.get("thread_id")
            
            # Format input for the agent - prioritize string input for schema compatibility
            if isinstance(input_data, str):
                # Direct string input - preferred format for schema compatibility
                formatted_input = {"messages": [("user", input_data)]}
            elif isinstance(input_data, dict) and "messages" in input_data:
                # Process structured messages
                if isinstance(input_data["messages"], list):
                    messages_list = []
                    for msg in input_data["messages"]:
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            messages_list.append((msg["role"], msg["content"]))
                        elif isinstance(msg, str):
                            messages_list.append(("user", msg))
                    formatted_input = {"messages": messages_list}
                elif isinstance(input_data["messages"], str):
                    formatted_input = {"messages": [("user", input_data["messages"])]}
                else:
                    formatted_input = input_data
            else:
                # Default case - convert to string
                user_content = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data)
                formatted_input = {"messages": [("user", user_content)]}
            
            # Add thread tracking metadata if available
            if thread_id:
                formatted_input["thread_id"] = thread_id
                formatted_input["session_id"] = thread_id
                formatted_input["metadata"] = {
                    "thread_id": thread_id,
                    "session_id": thread_id
                }
                
                # Create config with thread tracking info
                config = {
                    "configurable": {"thread_id": thread_id},
                    "metadata": {
                        "thread_id": thread_id,
                        "session_id": thread_id
                    }
                }
                
                # Get thread-specific agent
                thread_agent = get_or_create_agent(thread_id)
                
                # Process through thread-aware agent
                response = await thread_agent.ainvoke(formatted_input, config)
            else:
                # Process through default agent for stateless requests
                response = await default_agent.ainvoke(formatted_input)
            
            # Process the response
            if hasattr(response, "content"):
                result = response.content
            elif isinstance(response, dict) and "messages" in response:
                messages = response.get("messages", [])
                if messages and len(messages) > 0:
                    last_message = messages[-1]
                    if isinstance(last_message, tuple) and len(last_message) >= 2:
                        result = str(last_message[1])
                    else:
                        result = str(messages)
                else:
                    result = str(response)
            elif isinstance(response, (list, tuple)):
                result = [str(r) for r in response]
            else:
                result = str(response)
            
            # Add thread_id to result if available
            if thread_id:
                if isinstance(result, dict):
                    result["thread_id"] = thread_id
                else:
                    result = {
                        "content": result,
                        "thread_id": thread_id
                    }
            
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(
                status_code=422,
                content={"error": str(e)}
            )

    return app

def _is_node_graph(spec):
    """Check if the given spec is a node graph."""
    if spec is None:
        return False
    file_path = spec.split(":")[0]
    return file_path.endswith(".py")
