# Rate Limiting Implementation

## Overview

To ensure responsible and ethical web scraping, rate limiting has been implemented across all pipelines that make web requests. This document describes the rate limiting approach and implementation.

## Rate Limiter Implementation

### Core Rate Limiter Class

A reusable `RateLimiter` class has been implemented in `utils/rate_limiter.py` with the following features:

- Token bucket algorithm to enforce request limits
- Configurable requests per time window
- Configurable minimum delay between individual requests
- Thread-safe implementation for concurrent use

```python
class RateLimiter:
    def __init__(
        self, 
        max_requests: int = 20, 
        time_window: int = 60,  # 60 seconds = 1 minute
        delay_between_requests: float = 0.5  # Half second delay between requests
    ):
        # Implementation...
    
    def wait_if_needed(self):
        # Implementation that enforces the rate limits
```

## Pipeline-Specific Rate Limiting

### HMRC Scraper

The HMRC scraper uses two different rate limiters:

1. `api_rate_limiter`: For GOV.UK API requests (20 requests per second)
2. `web_rate_limiter`: For general web page scraping (10 requests per second)

These rate limits ensure we don't overload the GOV.UK servers while still allowing for efficient data collection.

### Bailii Scraper

The Bailii scraper uses a more conservative rate limiter:

- 10 requests per minute with a 1-second delay between requests

For housing-specific Bailii scraping, an even more conservative approach is used:

- 5 requests per minute with a 2-second delay between requests

### Other Web Requests

Other utilities that make web requests (like mdgen.py) implement their own rate limiting using a similar approach:

- Github API requests: 15 requests per minute

## Integration Points

Rate limiting has been integrated at all points where HTTP requests are made:

1. Discovery phases (search API requests)
2. Content extraction (individual document downloads)
3. Form and manual discovery
4. External API interactions

## Best Practices

When developing new scrapers or modifying existing ones:

1. Always use the appropriate rate limiter for each request
2. Be more conservative with public-facing APIs
3. Consider the target site's terms of service and robot.txt rules
4. Monitor for HTTP 429 responses and adjust rate limits accordingly
5. Implement graceful fallbacks when rate limits are hit

## Example Usage

```python
# Initialize rate limiter
rate_limiter = RateLimiter(max_requests=10, time_window=60)

# In request function
def make_request(url):
    # Apply rate limiting before making the request
    rate_limiter.wait_if_needed()
    
    # Make the request
    response = requests.get(url)
    # ...
```
