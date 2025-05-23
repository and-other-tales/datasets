#!/usr/bin/env python3
"""
Compliant UK Legislation Downloader

This script follows the official legislation.gov.uk API documentation
from https://legislation.github.io/data-documentation/

Key features:
- Proper rate limiting (3,000 requests per 5 minutes)
- Correct User-Agent header
- Uses official API endpoints (/data.xml, /data.feed)
- Handles 202 responses for dynamic content
- Uses Atom feeds for efficient discovery
"""

import os
import sys
import time
import json
import requests
from urllib.parse import urljoin, urlparse
from pathlib import Path
import logging
from typing import List, Dict, Set, Optional
import xml.etree.ElementTree as ET
from collections import deque
import threading
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compliant_legislation_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter implementing the official 3,000 requests per 5 minutes limit"""
    
    def __init__(self, max_requests: int = 3000, time_window: int = 300):
        self.max_requests = max_requests
        self.time_window = time_window  # 5 minutes = 300 seconds
        self.requests = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we would exceed the rate limit"""
        with self.lock:
            now = time.time()
            
            # Remove requests older than time_window
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()
            
            # If we're at the limit, wait until we can make another request
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.time_window - now + 1
                if sleep_time > 0:
                    logger.info(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                    # Clean up again after waiting
                    while self.requests and self.requests[0] < time.time() - self.time_window:
                        self.requests.popleft()
            
            # Record this request
            self.requests.append(now)

class CompliantLegislationDownloader:
    def __init__(self, output_dir: str = "compliant_legislation"):
        self.base_url = "https://www.legislation.gov.uk"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories following CLML structure
        self.xml_dir = self.output_dir / "xml"
        self.metadata_dir = self.output_dir / "metadata"
        self.feeds_dir = self.output_dir / "feeds"
        
        for dir_path in [self.xml_dir, self.metadata_dir, self.feeds_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter()
        
        # Session with proper headers as per documentation
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CompliantLegislationDownloader/1.0 (Educational/Research; contact@example.com)'
        })
        
        self.discovered_urls: Set[str] = set()
        self.downloaded_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        
        # Official legislation types from API documentation
        self.legislation_types = [
            'ukpga',   # UK Public General Acts
            'ukla',    # UK Local Acts
            'uksi',    # UK Statutory Instruments
            'ukdsi',   # UK Draft Statutory Instruments
            'asp',     # Acts of the Scottish Parliament
            'ssi',     # Scottish Statutory Instruments
            'asc',     # Acts of Senedd Cymru
            'wsi',     # Welsh Statutory Instruments
            'anaw',    # Acts of the National Assembly for Wales
            'mwa',     # Measures of the National Assembly for Wales
            'nia',     # Northern Ireland Acts
            'nisr',    # Northern Ireland Statutory Rules
            'ukmo',    # UK Ministerial Orders
            'ukdmo',   # UK Draft Ministerial Orders
            'uksro',   # UK Statutory Rules and Orders
            'nisro',   # Northern Ireland Statutory Rules and Orders
            'ukci',    # UK Church Instruments
            'ukcm',    # UK Church Measures
        ]
    
    def make_request(self, url: str, **kwargs) -> Optional[requests.Response]:
        """Make a rate-limited request following API guidelines"""
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(url, timeout=30, **kwargs)
            
            # Handle different response codes as per documentation
            if response.status_code == 200:
                return response
            elif response.status_code == 202:
                # Dynamically generated resource - may need to wait and retry
                logger.info(f"Resource being generated for {url}, waiting...")
                time.sleep(5)
                self.rate_limiter.wait_if_needed()
                retry_response = self.session.get(url, timeout=30, **kwargs)
                return retry_response if retry_response.status_code == 200 else None
            elif response.status_code == 403:
                logger.error(f"Rate limit hit (403) for {url}")
                time.sleep(60)  # Wait a minute before continuing
                return None
            elif response.status_code == 404:
                logger.debug(f"Resource not found: {url}")
                return None
            else:
                logger.warning(f"Unexpected status {response.status_code} for {url}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def discover_legislation_via_feeds(self) -> Set[str]:
        """Discover legislation using official Atom feeds"""
        logger.info("Discovering legislation via official Atom feeds...")
        
        discovered_urls = set()
        
        for leg_type in self.legislation_types:
            logger.info(f"Discovering {leg_type} legislation via feed...")
            
            # Get the main feed for this legislation type
            feed_url = f"{self.base_url}/{leg_type}/data.feed"
            
            response = self.make_request(feed_url)
            if not response:
                continue
            
            try:
                # Parse Atom feed
                root = ET.fromstring(response.content)
                
                # Extract namespace
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                
                # Find all entry links
                entries = root.findall('.//atom:entry', ns)
                
                for entry in entries:
                    link_elem = entry.find('atom:link[@rel="alternate"]', ns)
                    if link_elem is not None:
                        href = link_elem.get('href')
                        if href:
                            full_url = urljoin(self.base_url, href)
                            discovered_urls.add(full_url)
                
                logger.info(f"Found {len(entries)} items in {leg_type} feed")
                
                # Save the feed for reference
                feed_file = self.feeds_dir / f"{leg_type}_feed.xml"
                with open(feed_file, 'wb') as f:
                    f.write(response.content)
                
            except ET.ParseError as e:
                logger.error(f"Error parsing feed for {leg_type}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing {leg_type} feed: {e}")
                continue
        
        logger.info(f"Total legislation items discovered: {len(discovered_urls)}")
        self.discovered_urls.update(discovered_urls)
        
        return discovered_urls
    
    def discover_by_year_browsing(self) -> Set[str]:
        """Supplementary discovery by browsing year pages"""
        logger.info("Supplementary discovery by year browsing...")
        
        discovered_urls = set()
        current_year = datetime.now().year
        
        for leg_type in self.legislation_types:
            # Focus on recent years to avoid overwhelming the API
            for year in range(max(1980, current_year - 50), current_year + 1):
                year_feed_url = f"{self.base_url}/{leg_type}/{year}/data.feed"
                
                response = self.make_request(year_feed_url)
                if not response:
                    continue
                
                try:
                    root = ET.fromstring(response.content)
                    ns = {'atom': 'http://www.w3.org/2005/Atom'}
                    
                    entries = root.findall('.//atom:entry', ns)
                    for entry in entries:
                        link_elem = entry.find('atom:link[@rel="alternate"]', ns)
                        if link_elem is not None:
                            href = link_elem.get('href')
                            if href:
                                full_url = urljoin(self.base_url, href)
                                discovered_urls.add(full_url)
                    
                    if entries:
                        logger.info(f"Found {len(entries)} items for {leg_type}/{year}")
                
                except ET.ParseError:
                    # Year may not have a valid feed
                    continue
                except Exception as e:
                    logger.warning(f"Error processing {leg_type}/{year}: {e}")
                    continue
        
        self.discovered_urls.update(discovered_urls)
        return discovered_urls
    
    def download_legislation_item(self, url: str) -> bool:
        """Download a single legislation item using the official API"""
        try:
            if url in self.downloaded_urls:
                return True
            
            # Extract ID from URL for filename
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.strip('/').split('/')
            item_id = '_'.join(path_parts)
            
            logger.info(f"Downloading {item_id}...")
            
            # Get XML data using the official /data.xml endpoint
            xml_url = f"{url}/data.xml"
            xml_response = self.make_request(xml_url)
            
            if xml_response and xml_response.status_code == 200:
                # Save XML content
                xml_file = self.xml_dir / f"{item_id}.xml"
                with open(xml_file, 'wb') as f:
                    f.write(xml_response.content)
                
                # Extract and save metadata from XML
                try:
                    root = ET.fromstring(xml_response.content)
                    metadata = self._extract_metadata_from_xml(root, url)
                    
                    metadata_file = self.metadata_dir / f"{item_id}.json"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                except ET.ParseError as e:
                    logger.warning(f"Could not parse XML metadata for {item_id}: {e}")
                
                self.downloaded_urls.add(url)
                logger.info(f"Successfully downloaded {item_id}")
                return True
            
            else:
                logger.warning(f"Could not download XML for {item_id}")
                self.failed_urls.add(url)
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            self.failed_urls.add(url)
            return False
    
    def _extract_metadata_from_xml(self, root: ET.Element, url: str) -> Dict:
        """Extract metadata from CLML XML"""
        metadata = {
            'url': url,
            'download_timestamp': datetime.now().isoformat(),
            'title': '',
            'type': '',
            'year': '',
            'number': '',
            'subjects': [],
            'department': '',
            'made_date': '',
            'laid_date': '',
            'coming_into_force_date': ''
        }
        
        try:
            # Extract basic information (CLML format)
            # This is a simplified extraction - full CLML parsing would be more complex
            
            # Try to find title
            title_elem = root.find('.//dc:title', {'dc': 'http://purl.org/dc/elements/1.1/'})
            if title_elem is not None and title_elem.text:
                metadata['title'] = title_elem.text.strip()
            
            # Try to find type
            type_elem = root.find('.//dc:type', {'dc': 'http://purl.org/dc/elements/1.1/'})
            if type_elem is not None and type_elem.text:
                metadata['type'] = type_elem.text.strip()
            
            # Extract year and number from URL if not found in XML
            path_parts = urlparse(url).path.strip('/').split('/')
            if len(path_parts) >= 3:
                metadata['type'] = path_parts[0]
                metadata['year'] = path_parts[1]
                if len(path_parts) >= 3:
                    metadata['number'] = path_parts[2]
            
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
        
        return metadata
    
    def download_all_discovered(self, max_items: Optional[int] = None):
        """Download all discovered legislation"""
        if not self.discovered_urls:
            logger.warning("No URLs discovered yet. Running discovery first...")
            self.discover_legislation_via_feeds()
        
        urls_to_download = list(self.discovered_urls - self.downloaded_urls)
        
        if max_items:
            urls_to_download = urls_to_download[:max_items]
        
        total_items = len(urls_to_download)
        logger.info(f"Starting download of {total_items} legislation items...")
        
        for i, url in enumerate(urls_to_download, 1):
            logger.info(f"Progress: {i}/{total_items}")
            
            success = self.download_legislation_item(url)
            
            if success:
                logger.info(f"Successfully downloaded item {i}/{total_items}")
            else:
                logger.warning(f"Failed to download item {i}/{total_items}")
            
            # Save progress periodically
            if i % 100 == 0:
                self.save_progress()
        
        logger.info(f"Download complete. Success: {len(self.downloaded_urls)}, Failed: {len(self.failed_urls)}")
    
    def save_progress(self):
        """Save current progress"""
        progress = {
            'discovered_urls': list(self.discovered_urls),
            'downloaded_urls': list(self.downloaded_urls),
            'failed_urls': list(self.failed_urls),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / "progress.json", 'w') as f:
            json.dump(progress, f, indent=2)
        
        logger.info(f"Progress saved: {len(self.discovered_urls)} discovered, "
                   f"{len(self.downloaded_urls)} downloaded, {len(self.failed_urls)} failed")
    
    def load_progress(self):
        """Load previous progress"""
        progress_file = self.output_dir / "progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                
                self.discovered_urls = set(progress.get('discovered_urls', []))
                self.downloaded_urls = set(progress.get('downloaded_urls', []))
                self.failed_urls = set(progress.get('failed_urls', []))
                
                logger.info(f"Loaded progress: {len(self.discovered_urls)} discovered, "
                           f"{len(self.downloaded_urls)} downloaded, {len(self.failed_urls)} failed")
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics"""
        summary = {
            'discovery': {
                'total_discovered': len(self.discovered_urls),
                'total_downloaded': len(self.downloaded_urls),
                'total_failed': len(self.failed_urls),
                'success_rate': len(self.downloaded_urls) / len(self.discovered_urls) if self.discovered_urls else 0
            },
            'files': {
                'xml_files': len(list(self.xml_dir.glob('*.xml'))),
                'metadata_files': len(list(self.metadata_dir.glob('*.json'))),
                'feed_files': len(list(self.feeds_dir.glob('*.xml')))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        with open(self.output_dir / "download_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    """Main function to run the compliant legislation downloader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download UK legislation using official API")
    parser.add_argument('--output-dir', default='compliant_legislation',
                       help='Directory to store legislation')
    parser.add_argument('--discover-only', action='store_true',
                       help='Only discover URLs, do not download')
    parser.add_argument('--max-items', type=int,
                       help='Maximum number of items to download')
    
    args = parser.parse_args()
    
    downloader = CompliantLegislationDownloader(args.output_dir)
    
    try:
        # Load previous progress
        downloader.load_progress()
        
        # Discover legislation
        logger.info("Starting legislation discovery...")
        downloader.discover_legislation_via_feeds()
        downloader.discover_by_year_browsing()
        
        if not args.discover_only:
            # Download all discovered legislation
            downloader.download_all_discovered(max_items=args.max_items)
        
        # Generate summary
        summary = downloader.generate_summary()
        
        print(f"\n=== COMPLIANT LEGISLATION DOWNLOAD SUMMARY ===")
        print(f"Total discovered: {summary['discovery']['total_discovered']}")
        print(f"Total downloaded: {summary['discovery']['total_downloaded']}")
        print(f"Success rate: {summary['discovery']['success_rate']:.2%}")
        print(f"XML files: {summary['files']['xml_files']}")
        print(f"Metadata files: {summary['files']['metadata_files']}")
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        downloader.save_progress()
    except Exception as e:
        logger.error(f"Download failed: {e}")
        downloader.save_progress()
        sys.exit(1)

if __name__ == "__main__":
    main()