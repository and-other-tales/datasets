#!/usr/bin/env python3
"""
Test script for improved HMRC scraper curses interface
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.hmrc_curses_wrapper import run_hmrc_scraper_with_curses
from pipelines.hmrc_scraper import HMRCScraper

def test_hmrc_scraper():
    """Test HMRC scraper with improved curses interface"""
    scraper = HMRCScraper("test_hmrc_output")
    
    # Run discovery only to test formatting
    print("Starting HMRC comprehensive discovery...")
    scraper.run_comprehensive_discovery()
    
    # Download just a few documents to test
    print("Downloading sample documents...")
    scraper.download_all_documents(max_documents=5)
    
    # Generate summary
    summary = scraper.generate_summary()
    
    print(f"\nTest completed!")
    print(f"Discovered: {summary['total_discovered']} documents")
    print(f"Downloaded: {summary['total_downloaded']} documents")
    
    return summary

if __name__ == "__main__":
    # Run with improved curses interface
    run_hmrc_scraper_with_curses(test_hmrc_scraper)