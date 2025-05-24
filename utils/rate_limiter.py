#!/usr/bin/env python3
"""
Rate Limiter for Web Scraping

This module provides a reusable rate limiting class for web scraping operations.
"""

import time
import logging
import threading
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    A flexible rate limiter for web scraping.
    
    This class implements a token bucket algorithm to enforce rate limits:
    - Limits requests to a specified maximum per time window
    - Provides configurable delay between individual requests
    - Thread-safe implementation for concurrent scrapers
    """
    
    def __init__(
        self, 
        max_requests: int = 20, 
        time_window: int = 60,  # 60 seconds = 1 minute
        delay_between_requests: float = 0.5  # Half second delay between requests
    ):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
            delay_between_requests: Minimum delay between consecutive requests in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.delay_between_requests = delay_between_requests
        self.requests = deque()
        self.lock = threading.Lock()
        self.last_request_time = 0
        
        logger.info(f"Rate limiter initialized: {max_requests} requests per {time_window}s with {delay_between_requests}s delay")
    
    def wait_if_needed(self):
        """
        Wait if necessary to comply with the rate limit.
        
        This method:
        1. Enforces the minimum delay between requests
        2. Ensures we don't exceed the maximum requests per time window
        """
        with self.lock:
            now = time.time()
            
            # First, ensure minimum delay between requests
            time_since_last_request = now - self.last_request_time
            if time_since_last_request < self.delay_between_requests:
                delay = self.delay_between_requests - time_since_last_request
                time.sleep(delay)
                now = time.time()  # Update current time after sleeping
            
            # Then, handle the requests per time window limit
            
            # Remove requests older than time_window
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()
            
            # If we're at the limit, wait until we can make another request
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.time_window - now + 0.1
                if sleep_time > 0:
                    logger.info(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                    # Clean up again after waiting
                    now = time.time()
                    while self.requests and self.requests[0] < now - self.time_window:
                        self.requests.popleft()
            
            # Record this request
            self.requests.append(now)
            self.last_request_time = now

    def log_status(self):
        """Log the current status of the rate limiter"""
        with self.lock:
            now = time.time()
            # Remove expired timestamps
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()
                
            request_count = len(self.requests)
            capacity_pct = (request_count / self.max_requests) * 100
            
            logger.debug(f"Rate limiter status: {request_count}/{self.max_requests} requests " 
                         f"({capacity_pct:.1f}% capacity used)")
