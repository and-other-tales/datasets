#!/usr/bin/env python3
"""Test HMRC discovery directly"""

import logging
from pipelines.hmrc_scraper import HMRCScraper

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_discovery():
    scraper = HMRCScraper("test_hmrc_output")
    
    # Run discovery
    scraper.run_comprehensive_discovery()
    
    print(f"\nDiscovered {len(scraper.discovered_urls)} documents")
    
    # Show first 10 URLs
    if scraper.discovered_urls:
        print("\nFirst 10 discovered URLs:")
        for i, url in enumerate(list(scraper.discovered_urls)[:10], 1):
            print(f"{i}. {url}")
    else:
        print("\nNo URLs discovered!")

if __name__ == "__main__":
    test_discovery()