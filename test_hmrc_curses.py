#!/usr/bin/env python3
"""
Test script to verify HMRC scraper curses formatting fixes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.curses_pipeline_wrapper import run_pipeline_with_curses
from pipelines.hmrc_scraper import HMRCScraper

def test_hmrc_scraper():
    """Test HMRC scraper with limited documents to verify formatting"""
    scraper = HMRCScraper("test_hmrc_output")
    
    # Run discovery only to test formatting
    scraper.run_comprehensive_discovery()
    
    # Download just a few documents to test
    scraper.download_all_documents(max_documents=3)
    
    return "Test completed successfully"

if __name__ == "__main__":
    # Run with curses interface
    run_pipeline_with_curses("HMRC Scraper Test", test_hmrc_scraper)