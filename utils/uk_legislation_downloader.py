#!/usr/bin/env python3
"""
UK Legislation Downloader

This script systematically downloads all UK legislation from legislation.gov.uk
using their API endpoints. It fetches a complete list of legislation and downloads
the full text content of each piece.
"""

import os
import sys
import time
import json
import requests
from urllib.parse import urljoin, urlparse
from pathlib import Path
import logging
from typing import List, Dict, Set
import xml.etree.ElementTree as ET

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legislation_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class UKLegislationDownloader:
    def __init__(self, output_dir: str = "uk_legislation"):
        self.base_url = "https://www.legislation.gov.uk"
        self.api_base = "https://www.legislation.gov.uk"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types
        self.text_dir = self.output_dir / "text"
        self.xml_dir = self.output_dir / "xml"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.text_dir, self.xml_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UK-Legislation-Downloader/1.0 (Educational/Research Purpose)'
        })
        
        self.downloaded_items = set()
        self.failed_items = set()
        self.legislation_list = []
        
    def get_legislation_list(self) -> List[Dict]:
        """Fetch complete list of UK legislation using search API"""
        logger.info("Fetching complete legislation list...")
        
        legislation_types = [
            'ukpga',  # UK Public General Acts
            'ukla',   # UK Local Acts
            'uksi',   # UK Statutory Instruments
            'ukdsi',  # UK Draft Statutory Instruments
            'asp',    # Acts of the Scottish Parliament
            'ssi',    # Scottish Statutory Instruments
            'asc',    # Acts of Senedd Cymru
            'wsi',    # Welsh Statutory Instruments
            'anaw',   # Acts of the National Assembly for Wales
            'mwa',    # Measures of the National Assembly for Wales
            'nia',    # Northern Ireland Acts
            'nisr',   # Northern Ireland Statutory Rules
            'ukmo',   # UK Ministerial Orders
            'ukdmo',  # UK Draft Ministerial Orders
            'uksro',  # UK Statutory Rules and Orders
            'nisro',  # Northern Ireland Statutory Rules and Orders
            'ukci',   # UK Church Instruments
            'ukcm',   # UK Church Measures
        ]
        
        all_legislation = []
        
        for leg_type in legislation_types:
            logger.info(f"Fetching {leg_type} legislation...")
            page = 1
            
            while True:
                try:
                    # Use search API with pagination
                    search_url = f"{self.api_base}/{leg_type}"
                    params = {
                        'page': page,
                        'view': 'plain'
                    }
                    
                    response = self.session.get(search_url, params=params, timeout=30)
                    response.raise_for_status()
                    
                    # Parse HTML to extract legislation items
                    legislation_items = self._parse_legislation_page(response.text, leg_type)
                    
                    if not legislation_items:
                        logger.info(f"No more items found for {leg_type} at page {page}")
                        break
                    
                    all_legislation.extend(legislation_items)
                    logger.info(f"Found {len(legislation_items)} items on page {page} for {leg_type}")
                    
                    page += 1
                    time.sleep(1)  # Rate limiting
                    
                except requests.RequestException as e:
                    logger.error(f"Error fetching {leg_type} page {page}: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error for {leg_type}: {e}")
                    break
        
        self.legislation_list = all_legislation
        logger.info(f"Total legislation items found: {len(all_legislation)}")
        
        # Save the list for reference
        with open(self.output_dir / "legislation_list.json", 'w') as f:
            json.dump(all_legislation, f, indent=2)
        
        return all_legislation
    
    def _parse_legislation_page(self, html_content: str, leg_type: str) -> List[Dict]:
        """Parse HTML page to extract legislation items"""
        items = []
        
        # This is a simplified parser - in reality you'd need more robust HTML parsing
        # For now, let's try to extract using the data API approach
        try:
            # Try to get XML data instead
            xml_url = html_content.replace('.html', '/data.xml') if '.html' in html_content else None
            # This is a placeholder - the actual implementation would need proper HTML parsing
            # using BeautifulSoup or similar library
            pass
        except Exception as e:
            logger.warning(f"Could not parse page for {leg_type}: {e}")
        
        return items
    
    def download_legislation_item(self, item: Dict) -> bool:
        """Download a single piece of legislation"""
        try:
            item_id = item.get('id', 'unknown')
            item_url = item.get('url', '')
            
            if not item_url:
                logger.warning(f"No URL found for item {item_id}")
                return False
            
            logger.info(f"Downloading {item_id}...")
            
            # Download XML version
            xml_url = f"{item_url}/data.xml"
            xml_response = self.session.get(xml_url, timeout=30)
            
            if xml_response.status_code == 200:
                xml_file = self.xml_dir / f"{item_id}.xml"
                with open(xml_file, 'wb') as f:
                    f.write(xml_response.content)
                logger.info(f"Downloaded XML for {item_id}")
            
            # Download plain text version
            text_url = f"{item_url}/data.txt"
            text_response = self.session.get(text_url, timeout=30)
            
            if text_response.status_code == 200:
                text_file = self.text_dir / f"{item_id}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text_response.text)
                logger.info(f"Downloaded text for {item_id}")
            
            # Save metadata
            metadata_file = self.metadata_dir / f"{item_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(item, f, indent=2)
            
            self.downloaded_items.add(item_id)
            return True
            
        except requests.RequestException as e:
            logger.error(f"Network error downloading {item_id}: {e}")
            self.failed_items.add(item_id)
            return False
        except Exception as e:
            logger.error(f"Error downloading {item_id}: {e}")
            self.failed_items.add(item_id)
            return False
    
    def download_all_legislation(self):
        """Download all legislation systematically"""
        if not self.legislation_list:
            self.get_legislation_list()
        
        total_items = len(self.legislation_list)
        logger.info(f"Starting download of {total_items} legislation items...")
        
        for i, item in enumerate(self.legislation_list, 1):
            logger.info(f"Progress: {i}/{total_items}")
            
            success = self.download_legislation_item(item)
            
            if success:
                logger.info(f"Successfully downloaded item {i}/{total_items}")
            else:
                logger.warning(f"Failed to download item {i}/{total_items}")
            
            # Rate limiting
            time.sleep(2)
            
            # Save progress periodically
            if i % 100 == 0:
                self.save_progress()
        
        logger.info(f"Download complete. Success: {len(self.downloaded_items)}, Failed: {len(self.failed_items)}")
    
    def verify_downloads(self) -> Dict[str, int]:
        """Verify all files were downloaded correctly"""
        logger.info("Starting verification round...")
        
        stats = {
            'expected': len(self.legislation_list),
            'downloaded': 0,
            'missing': 0,
            'corrupted': 0
        }
        
        missing_items = []
        
        for item in self.legislation_list:
            item_id = item.get('id', 'unknown')
            
            # Check if files exist
            xml_file = self.xml_dir / f"{item_id}.xml"
            text_file = self.text_dir / f"{item_id}.txt"
            metadata_file = self.metadata_dir / f"{item_id}.json"
            
            if not (xml_file.exists() or text_file.exists()):
                missing_items.append(item)
                stats['missing'] += 1
                logger.warning(f"Missing files for {item_id}")
            else:
                # Basic corruption check
                try:
                    if xml_file.exists():
                        with open(xml_file, 'r') as f:
                            content = f.read(100)  # Check first 100 chars
                        if len(content) == 0:
                            stats['corrupted'] += 1
                            logger.warning(f"Empty XML file for {item_id}")
                            continue
                    
                    stats['downloaded'] += 1
                    
                except Exception as e:
                    stats['corrupted'] += 1
                    logger.warning(f"Corrupted file for {item_id}: {e}")
        
        # Retry missing items
        if missing_items:
            logger.info(f"Retrying {len(missing_items)} missing items...")
            for item in missing_items:
                self.download_legislation_item(item)
                time.sleep(1)
        
        # Final verification
        final_stats = self._count_files()
        logger.info(f"Verification complete: {final_stats}")
        
        return final_stats
    
    def _count_files(self) -> Dict[str, int]:
        """Count downloaded files"""
        return {
            'xml_files': len(list(self.xml_dir.glob('*.xml'))),
            'text_files': len(list(self.text_dir.glob('*.txt'))),
            'metadata_files': len(list(self.metadata_dir.glob('*.json')))
        }
    
    def save_progress(self):
        """Save current progress"""
        progress = {
            'downloaded_items': list(self.downloaded_items),
            'failed_items': list(self.failed_items),
            'total_expected': len(self.legislation_list)
        }
        
        with open(self.output_dir / "progress.json", 'w') as f:
            json.dump(progress, f, indent=2)
    
    def load_progress(self):
        """Load previous progress"""
        progress_file = self.output_dir / "progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            
            self.downloaded_items = set(progress.get('downloaded_items', []))
            self.failed_items = set(progress.get('failed_items', []))
            logger.info(f"Loaded progress: {len(self.downloaded_items)} downloaded, {len(self.failed_items)} failed")

def main():
    """Main function to run the legislation downloader"""
    downloader = UKLegislationDownloader()
    
    try:
        # Load any previous progress
        downloader.load_progress()
        
        # Get legislation list
        downloader.get_legislation_list()
        
        # Download all legislation
        downloader.download_all_legislation()
        
        # Verify downloads
        stats = downloader.verify_downloads()
        
        logger.info("Script completed successfully!")
        logger.info(f"Final statistics: {stats}")
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        downloader.save_progress()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        downloader.save_progress()
        sys.exit(1)

if __name__ == "__main__":
    main()