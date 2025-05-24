#!/usr/bin/env python3
"""
Simple test script to demonstrate rate limiting functionality
"""

import sys
import time
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from utils.rate_limiter import RateLimiter

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rate_limiter():
    """Test the rate limiter with different configurations"""
    
    logger.info("=== RATE LIMITER DEMONSTRATION ===")
    
    # Test 1: Fast rate limiter (10 requests per second)
    fast_limiter = RateLimiter(max_requests=10, time_window=1, delay_between_requests=0.05)
    
    logger.info("Test 1: Fast rate limiter (10 requests per second)")
    start_time = time.time()
    for i in range(20):
        fast_limiter.wait_if_needed()
        logger.info(f"Fast request {i+1} processed at {time.time() - start_time:.3f}s")
    
    duration = time.time() - start_time
    logger.info(f"20 requests with fast rate limiting took {duration:.2f} seconds")
    logger.info(f"Effective rate: {20/duration:.2f} requests/second (limit: 10 req/sec)")
    
    # Brief pause between tests
    time.sleep(2)
    
    # Test 2: Slow rate limiter (2 requests per second)
    slow_limiter = RateLimiter(max_requests=2, time_window=1, delay_between_requests=0.1)
    
    logger.info("\nTest 2: Slow rate limiter (2 requests per second)")
    start_time = time.time()
    for i in range(10):
        slow_limiter.wait_if_needed()
        logger.info(f"Slow request {i+1} processed at {time.time() - start_time:.3f}s")
    
    duration = time.time() - start_time
    logger.info(f"10 requests with slow rate limiting took {duration:.2f} seconds")
    logger.info(f"Effective rate: {10/duration:.2f} requests/second (limit: 2 req/sec)")
    
    # Test 3: Burst handling (5 requests per 2 seconds)
    burst_limiter = RateLimiter(max_requests=5, time_window=2, delay_between_requests=0.1)
    
    logger.info("\nTest 3: Burst handling (5 requests per 2 seconds)")
    logger.info("First burst of 5 requests:")
    
    start_time = time.time()
    for i in range(5):
        burst_limiter.wait_if_needed()
        logger.info(f"Burst 1 request {i+1} processed at {time.time() - start_time:.3f}s")
    
    logger.info("\nSecond burst of 5 requests (should be delayed):")
    for i in range(5):
        burst_limiter.wait_if_needed()
        logger.info(f"Burst 2 request {i+1} processed at {time.time() - start_time:.3f}s")
    
    duration = time.time() - start_time
    logger.info(f"10 requests with burst handling took {duration:.2f} seconds")
    logger.info(f"Expected time: ~4 seconds (2 seconds per 5 requests)")
    
    logger.info("\n=== RATE LIMITER DEMONSTRATION COMPLETE ===")

if __name__ == "__main__":
    test_rate_limiter()
