#!/usr/bin/env python3
"""
Comprehensive test for both rate limiting and curses interface fixes
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from utils.hmrc_curses_wrapper import run_hmrc_scraper_with_curses
from utils.rate_limiter import RateLimiter
from pipelines.hmrc_scraper import HMRCScraper

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rate_limiting_and_curses():
    """Test both rate limiting and curses interface together"""
    output_dir = "test_combined_fixes"
    
    # Create a test logger to verify rate limiting and display
    test_logger = logging.getLogger("test_combined")
    test_logger.setLevel(logging.INFO)
    
    # Test messages with special characters to verify curses handling
    test_messages = [
        "Regular ASCII message",
        "Message with 'quotes' and symbols: ©®™",
        "Emoji test: 🔍 📊 🚀 ✅ ❌",
        "Long line test: " + "-" * 100,
        "Multi-line\ntest\nmessage",
        "Error test",
        "Warning test",
        "Success test"
    ]
    
    # Create a rate limiter for testing
    test_limiter = RateLimiter(max_requests=5, time_window=1, delay_between_requests=0.1)
    
    logger.info("Starting combined rate limiting and curses test")
    
    # Create scraper with limited options
    scraper = HMRCScraper(output_dir)
    
    # Test function to be run with curses wrapper
    def run_test():
        logger.info("=== RATE LIMITER TEST ===")
        
        # Test rate limiting
        start_time = time.time()
        for i in range(10):
            test_limiter.wait_if_needed()
            logger.info(f"Request {i+1} processed at {time.time() - start_time:.2f}s")
        
        duration = time.time() - start_time
        logger.info(f"10 requests with rate limiting took {duration:.2f} seconds")
        logger.info(f"Effective rate: {10/duration:.2f} requests/second (limit: 5 req/sec)")
        
        # Test curses display with special characters
        logger.info("=== CURSES DISPLAY TEST ===")
        for msg in test_messages:
            if "Error" in msg:
                logger.error(msg)
            elif "Warning" in msg:
                logger.warning(msg)
            else:
                logger.info(msg)
            time.sleep(0.5)  # Pause to see each message
        
        # Run a small portion of the HMRC scraper to test both features together
        logger.info("=== HMRC SCRAPER TEST ===")
        urls = scraper.discover_via_search_api(max_batches=2)
        if urls:
            logger.info(f"Discovered {len(urls)} documents")
            scraper.download_all_documents(max_documents=3)
            
            # Generate summary
            summary = scraper.generate_summary()
            logger.info(f"Test completed successfully!")
            logger.info(f"Discovered: {summary['total_discovered']} documents")
            logger.info(f"Downloaded: {summary['total_downloaded']} documents")
        
        return {
            'status': 'success',
            'message': 'Rate limiting and curses interface test completed successfully'
        }
    
    # Run with curses wrapper
    return run_hmrc_scraper_with_curses(run_test)

if __name__ == "__main__":
    test_rate_limiting_and_curses()
