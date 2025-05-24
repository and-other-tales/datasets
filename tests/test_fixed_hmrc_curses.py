#!/usr/bin/env python3
"""
Test script for fixed HMRC scraper curses interface
Run this to test the fixed curses implementation
"""

import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.hmrc_curses_wrapper_fixed import run_hmrc_scraper_with_curses
from pipelines.hmrc_scraper import HMRCScraper

def test_fixed_hmrc_curses():
    """Test HMRC scraper with fixed curses interface"""
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create a test output directory
    output_dir = "test_hmrc_output_fixed"
    
    logger.info("Starting HMRC test with fixed curses interface")
    
    # Create scraper with limited options
    scraper = HMRCScraper(output_dir)
    
    # Run discovery with limited batches
    logger.info("Running limited discovery phase...")
    urls = scraper.discover_via_search_api(max_batches=5)
    scraper.discovered_urls = urls
    
    logger.info(f"Discovered {len(urls)} documents")
    
    # Download a small sample of documents
    if urls:
        logger.info("Downloading sample documents...")
        scraper.download_all_documents(max_documents=10)
        
        # Generate summary
        summary = scraper.generate_summary()
        
        logger.info(f"Test completed successfully!")
        logger.info(f"Discovered: {summary['total_discovered']} documents")
        logger.info(f"Downloaded: {summary['total_downloaded']} documents")
        
        return {
            'status': 'success',
            'discovered': summary['total_discovered'],
            'downloaded': summary['total_downloaded']
        }
    else:
        logger.error("Discovery failed - no documents found")
        return {
            'status': 'error',
            'error': 'No documents discovered'
        }

if __name__ == "__main__":
    # Run with fixed curses interface
    run_hmrc_scraper_with_curses(test_fixed_hmrc_curses)
