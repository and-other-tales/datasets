#!/usr/bin/env python3
"""
Script to fetch and analyze LangGraph documentation from specified URLs.
"""
import requests
from bs4 import BeautifulSoup
import json
import sys

URLS = [
    "https://langchain-ai.github.io/langgraph/cloud/how-tos/iterate_graph_studio/",
    "https://langchain-ai.github.io/langgraph/how-tos/configuration/",
    "https://langchain-ai.github.io/langgraph/cloud/reference/cli/",
    "https://langchain-ai.github.io/langgraph/reference/graphs/",
    "https://langchain-ai.github.io/langgraph/reference/checkpoints/",
    "https://langchain-ai.github.io/langgraph/agents/overview/#high-level-building-blocks",
    "https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/persistence_postgres.ipynb"
]

def get_text_from_url(url):
    """Fetch and extract text content from a URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        if "github.com" in url:
            # For GitHub, extract the raw content or notebook JSON
            if url.endswith(".ipynb"):
                soup = BeautifulSoup(response.text, 'html.parser')
                # Try to find the notebook content in the page
                script_tags = soup.find_all('script')
                for script in script_tags:
                    if script.string and "json" in script.get('type', ''):
                        try:
                            data = json.loads(script.string)
                            if isinstance(data, dict) and "payload" in data:
                                notebook_content = data.get("payload", {}).get("blob", {}).get("rawLines", [])
                                if notebook_content:
                                    return "\n".join(notebook_content)
                        except json.JSONDecodeError:
                            continue
                # If we couldn't extract it that way, return the HTML
                return soup.get_text(separator='\n', strip=True)
        
        # For documentation sites
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove navigation, headers, footers
        for element in soup.select('nav, header, footer, .sidebar, .navigation, .menu'):
            element.decompose()
        
        # Get the main content
        main_content = soup.select_one('main, .main-content, article, .markdown-body')
        if main_content:
            return main_content.get_text(separator='\n', strip=True)
        
        # If no main content found, get the body text
        return soup.get_text(separator='\n', strip=True)
    
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

# Fetch content from each URL
results = {}
for url in URLS:
    print(f"Fetching {url}...", file=sys.stderr)
    content = get_text_from_url(url)
    
    # Get the URL's "topic" from the last part of the path
    parts = url.rstrip('/').split('/')
    if '#' in parts[-1]:
        topic = parts[-1].split('#')[0]
    else:
        topic = parts[-1]
    
    # Clean up topic name
    if topic.endswith('.ipynb'):
        topic = topic.replace('.ipynb', '')
    topic = topic.replace('_', ' ').title()
    
    results[url] = {
        "topic": topic,
        "content": content[:10000]  # Limit content length
    }

# Output as JSON
print(json.dumps(results, indent=2))