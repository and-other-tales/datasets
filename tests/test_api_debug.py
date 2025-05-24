#!/usr/bin/env python3
"""Debug HMRC API discovery"""

import requests
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_discovery():
    base_url = "https://www.gov.uk"
    search_api_base = f"{base_url}/api/search.json"
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'HMRC-Documentation-Scraper/1.0 (Educational/Research Purpose)'
    })
    
    params = {
        'count': 10,
        'start': 0,
        'filter_organisations': 'hm-revenue-customs',
        'order': '-public_timestamp'
    }
    
    logger.info(f"Making API request to {search_api_base} with params: {params}")
    response = session.get(search_api_base, params=params, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    results = data.get('results', [])
    total_results = data.get('total', 0)
    
    logger.info(f"API returned {len(results)} results out of {total_results} total")
    
    # Log details of first few results
    for i, result in enumerate(results[:5]):
        title = result.get('title', '')
        format_type = result.get('format', '')
        link = result.get('link', '')
        description = result.get('description', '')
        
        print(f"\n--- Result {i+1} ---")
        print(f"Title: {title}")
        print(f"Format: {format_type}")
        print(f"Link: {link}")
        print(f"Description: {description[:100]}...")
        
        # Test quality check
        from pipelines.hmrc_scraper import HMRCScraper
        scraper = HMRCScraper()
        is_quality = scraper.is_high_quality_tax_content(title, description, format_type, result)
        print(f"Quality check passed: {is_quality}")

if __name__ == "__main__":
    test_discovery()