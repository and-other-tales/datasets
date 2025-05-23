#!/usr/bin/env python3
"""
ParaLlama UK Legislation Downloader

This script uses BeautifulSoup for proper HTML parsing and implements
a more robust approach to systematically download UK legislation for Legal Llama training.
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
import re
from bs4 import BeautifulSoup

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

class ParaLlamaUKLegislationDownloader:
    def __init__(self, output_dir: str = "uk_legislation"):
        self.base_url = "https://www.legislation.gov.uk"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.text_dir = self.output_dir / "text"
        self.xml_dir = self.output_dir / "xml"
        self.html_dir = self.output_dir / "html"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.text_dir, self.xml_dir, self.html_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UK-Legislation-Downloader/2.0 (Educational/Research Purpose)'
        })
        
        self.downloaded_items = set()
        self.failed_items = set()
        self.legislation_list = []
        
        # Known legislation types and their URL patterns
        self.legislation_types = {
            'ukpga': 'UK Public General Acts',
            'ukla': 'UK Local Acts', 
            'uksi': 'UK Statutory Instruments',
            'ukdsi': 'UK Draft Statutory Instruments',
            'asp': 'Acts of the Scottish Parliament',
            'ssi': 'Scottish Statutory Instruments',
            'asc': 'Acts of Senedd Cymru',
            'wsi': 'Welsh Statutory Instruments',
            'anaw': 'Acts of the National Assembly for Wales',
            'mwa': 'Measures of the National Assembly for Wales',
            'nia': 'Northern Ireland Acts',
            'nisr': 'Northern Ireland Statutory Rules',
            'ukmo': 'UK Ministerial Orders',
            'ukdmo': 'UK Draft Ministerial Orders',
            'uksro': 'UK Statutory Rules and Orders',
            'nisro': 'Northern Ireland Statutory Rules and Orders',
            'ukci': 'UK Church Instruments',
            'ukcm': 'UK Church Measures',
            'eur': 'EU Legislation',
            'eudr': 'EU Directives',
            'eurdn': 'EU Decisions'
        }
    
    def discover_legislation_systematically(self) -> List[Dict]:
        """Use multiple approaches to discover all legislation"""
        logger.info("Starting systematic legislation discovery...")
        
        all_legislation = []
        
        # Method 1: Browse by type and year
        for leg_type in self.legislation_types.keys():
            items = self._discover_by_type_and_year(leg_type)
            all_legislation.extend(items)
        
        # Method 2: Use search interface with broad queries
        search_items = self._discover_via_search()
        all_legislation.extend(search_items)
        
        # Remove duplicates
        seen_urls = set()
        unique_legislation = []
        for item in all_legislation:
            if item['url'] not in seen_urls:
                seen_urls.add(item['url'])
                unique_legislation.append(item)
        
        self.legislation_list = unique_legislation
        logger.info(f"Total unique legislation items discovered: {len(unique_legislation)}")
        
        # Save discovery results
        with open(self.output_dir / "discovered_legislation.json", 'w') as f:
            json.dump(unique_legislation, f, indent=2)
        
        return unique_legislation
    
    def _discover_by_type_and_year(self, leg_type: str) -> List[Dict]:
        """Discover legislation by browsing type pages systematically"""
        logger.info(f"Discovering {leg_type} legislation...")
        items = []
        
        # Try different year ranges
        current_year = 2024
        start_year = 1800  # Approximate start of modern legislation
        
        for year in range(start_year, current_year + 1):
            try:
                # Try the standard URL pattern
                url = f"{self.base_url}/{leg_type}/{year}"
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    year_items = self._parse_year_page(response.text, leg_type, year)
                    items.extend(year_items)
                    
                    if year_items:
                        logger.info(f"Found {len(year_items)} items for {leg_type}/{year}")
                    
                    time.sleep(0.5)  # Rate limiting
                
                elif response.status_code == 404:
                    # No legislation for this year/type combination
                    continue
                else:
                    logger.warning(f"Unexpected status {response.status_code} for {leg_type}/{year}")
                
            except requests.RequestException as e:
                logger.warning(f"Error fetching {leg_type}/{year}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error for {leg_type}/{year}: {e}")
                continue
        
        logger.info(f"Total items found for {leg_type}: {len(items)}")
        return items
    
    def _parse_year_page(self, html_content: str, leg_type: str, year: int) -> List[Dict]:
        """Parse a year page to extract legislation items"""
        items = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for legislation links - these vary by page structure
            # Common patterns for legislation links
            link_selectors = [
                'a[href*="/{}"]'.format(leg_type),
                '.LegislationItem a',
                '.item a',
                'a[href*="/{}/{}"]'.format(leg_type, year),
                '.title a'
            ]
            
            found_links = set()
            
            for selector in link_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href', '')
                    if href and leg_type in href and str(year) in href:
                        full_url = urljoin(self.base_url, href)
                        title = link.get_text(strip=True)
                        
                        if full_url not in found_links:
                            found_links.add(full_url)
                            
                            # Extract number from URL
                            number_match = re.search(r'/(\d+)(?:/|$)', href)
                            number = number_match.group(1) if number_match else 'unknown'
                            
                            items.append({
                                'id': f"{leg_type}_{year}_{number}",
                                'type': leg_type,
                                'year': year,
                                'number': number,
                                'title': title,
                                'url': full_url
                            })
            
            # Also try to find pagination
            next_links = soup.select('a[href*="page="], .next a, .pagination a')
            for next_link in next_links:
                href = next_link.get('href', '')
                if 'page=' in href and 'next' in next_link.get_text().lower():
                    # Handle pagination if needed
                    pass
                    
        except Exception as e:
            logger.warning(f"Error parsing year page for {leg_type}/{year}: {e}")
        
        return items
    
    def _discover_via_search(self) -> List[Dict]:
        """Use search interface to discover more legislation"""
        logger.info("Discovering legislation via search interface...")
        items = []
        
        # Try various search approaches
        search_params_list = [
            {'title': '*', 'year': ''},  # Wildcard search
            {'type': 'ukpga'},
            {'type': 'uksi'},
            # Add more search strategies as needed
        ]
        
        for search_params in search_params_list:
            try:
                search_url = f"{self.base_url}/search"
                response = self.session.get(search_url, params=search_params, timeout=30)
                
                if response.status_code == 200:
                    search_items = self._parse_search_results(response.text)
                    items.extend(search_items)
                    
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error in search discovery: {e}")
        
        return items
    
    def _parse_search_results(self, html_content: str) -> List[Dict]:
        """Parse search results page"""
        items = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for result items
            result_selectors = [
                '.searchResult',
                '.result',
                '.legislation-item',
                'li a[href*="/uk"]'
            ]
            
            for selector in result_selectors:
                results = soup.select(selector)
                for result in results:
                    link = result.find('a') if result.name != 'a' else result
                    if link:
                        href = link.get('href', '')
                        title = link.get_text(strip=True)
                        
                        if href:
                            full_url = urljoin(self.base_url, href)
                            
                            # Extract type and details from URL
                            url_parts = href.strip('/').split('/')
                            if len(url_parts) >= 2:
                                leg_type = url_parts[0]
                                year_num = url_parts[1] if len(url_parts) > 1 else 'unknown'
                                
                                items.append({
                                    'id': f"search_{leg_type}_{year_num}",
                                    'type': leg_type,
                                    'title': title,
                                    'url': full_url
                                })
        
        except Exception as e:
            logger.warning(f"Error parsing search results: {e}")
        
        return items
    
    def download_legislation_item(self, item: Dict) -> bool:
        """Download a single piece of legislation in multiple formats"""
        try:
            item_id = item.get('id', 'unknown')
            item_url = item.get('url', '')
            
            if not item_url:
                logger.warning(f"No URL found for item {item_id}")
                return False
            
            if item_id in self.downloaded_items:
                logger.info(f"Already downloaded {item_id}, skipping")
                return True
            
            logger.info(f"Downloading {item_id}...")
            
            success = False
            
            # Try to download XML version
            try:
                xml_url = f"{item_url}/data.xml"
                xml_response = self.session.get(xml_url, timeout=30)
                
                if xml_response.status_code == 200:
                    xml_file = self.xml_dir / f"{item_id}.xml"
                    with open(xml_file, 'wb') as f:
                        f.write(xml_response.content)
                    logger.info(f"Downloaded XML for {item_id}")
                    success = True
            except Exception as e:
                logger.warning(f"Could not download XML for {item_id}: {e}")
            
            # Try to download HTML version
            try:
                html_response = self.session.get(item_url, timeout=30)
                
                if html_response.status_code == 200:
                    html_file = self.html_dir / f"{item_id}.html"
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(html_response.text)
                    
                    # Extract text content from HTML
                    soup = BeautifulSoup(html_response.text, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Extract text
                    text_content = soup.get_text()
                    
                    # Clean up text
                    lines = (line.strip() for line in text_content.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text_content = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    text_file = self.text_dir / f"{item_id}.txt"
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    
                    logger.info(f"Downloaded HTML and extracted text for {item_id}")
                    success = True
                    
            except Exception as e:
                logger.warning(f"Could not download HTML for {item_id}: {e}")
            
            # Save metadata
            try:
                metadata_file = self.metadata_dir / f"{item_id}.json"
                with open(metadata_file, 'w') as f:
                    json.dump(item, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save metadata for {item_id}: {e}")
            
            if success:
                self.downloaded_items.add(item_id)
                return True
            else:
                self.failed_items.add(item_id)
                return False
            
        except Exception as e:
            logger.error(f"Error downloading {item_id}: {e}")
            self.failed_items.add(item_id)
            return False
    
    def download_all_legislation(self):
        """Download all discovered legislation"""
        if not self.legislation_list:
            self.discover_legislation_systematically()
        
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
            time.sleep(1)
            
            # Save progress periodically
            if i % 50 == 0:
                self.save_progress()
        
        logger.info(f"Download complete. Success: {len(self.downloaded_items)}, Failed: {len(self.failed_items)}")
    
    def verify_downloads(self) -> Dict[str, int]:
        """Verify all files were downloaded correctly and retry missing items"""
        logger.info("Starting verification round...")
        
        stats = {
            'expected': len(self.legislation_list),
            'xml_files': 0,
            'html_files': 0,
            'text_files': 0,
            'metadata_files': 0,
            'missing': 0,
            'corrupted': 0
        }
        
        missing_items = []
        
        for item in self.legislation_list:
            item_id = item.get('id', 'unknown')
            
            # Check if files exist
            xml_file = self.xml_dir / f"{item_id}.xml"
            html_file = self.html_dir / f"{item_id}.html"
            text_file = self.text_dir / f"{item_id}.txt"
            metadata_file = self.metadata_dir / f"{item_id}.json"
            
            files_exist = {
                'xml': xml_file.exists(),
                'html': html_file.exists(),
                'text': text_file.exists(),
                'metadata': metadata_file.exists()
            }
            
            # Count existing files
            for file_type, exists in files_exist.items():
                if exists:
                    stats[f'{file_type}_files'] += 1
            
            # Check if at least one content file exists
            if not (files_exist['xml'] or files_exist['html'] or files_exist['text']):
                missing_items.append(item)
                stats['missing'] += 1
                logger.warning(f"Missing files for {item_id}")
            else:
                # Basic corruption check
                try:
                    for file_type, file_path in [
                        ('xml', xml_file), 
                        ('html', html_file), 
                        ('text', text_file)
                    ]:
                        if file_path.exists():
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read(100)  # Check first 100 chars
                            if len(content) == 0:
                                stats['corrupted'] += 1
                                logger.warning(f"Empty {file_type} file for {item_id}")
                                break
                                
                except Exception as e:
                    stats['corrupted'] += 1
                    logger.warning(f"Corrupted file for {item_id}: {e}")
        
        # Retry missing items
        if missing_items:
            logger.info(f"Retrying {len(missing_items)} missing items...")
            for item in missing_items:
                self.download_legislation_item(item)
                time.sleep(1)
        
        # Final count
        final_stats = self._count_files()
        stats.update(final_stats)
        
        logger.info(f"Verification complete: {stats}")
        
        # Save verification report
        with open(self.output_dir / "verification_report.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def _count_files(self) -> Dict[str, int]:
        """Count downloaded files"""
        return {
            'xml_files': len(list(self.xml_dir.glob('*.xml'))),
            'html_files': len(list(self.html_dir.glob('*.html'))),
            'text_files': len(list(self.text_dir.glob('*.txt'))),
            'metadata_files': len(list(self.metadata_dir.glob('*.json')))
        }
    
    def save_progress(self):
        """Save current progress"""
        progress = {
            'downloaded_items': list(self.downloaded_items),
            'failed_items': list(self.failed_items),
            'total_expected': len(self.legislation_list),
            'timestamp': time.time()
        }
        
        with open(self.output_dir / "progress.json", 'w') as f:
            json.dump(progress, f, indent=2)
        
        logger.info(f"Progress saved: {len(self.downloaded_items)} downloaded, {len(self.failed_items)} failed")
    
    def load_progress(self):
        """Load previous progress"""
        progress_file = self.output_dir / "progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                
                self.downloaded_items = set(progress.get('downloaded_items', []))
                self.failed_items = set(progress.get('failed_items', []))
                logger.info(f"Loaded progress: {len(self.downloaded_items)} downloaded, {len(self.failed_items)} failed")
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")

def main():
    """Main function to run the improved legislation downloader"""
    downloader = ImprovedUKLegislationDownloader()
    
    try:
        # Load any previous progress
        downloader.load_progress()
        
        # Discover all legislation
        downloader.discover_legislation_systematically()
        
        # Download all legislation
        downloader.download_all_legislation()
        
        # Verify downloads
        stats = downloader.verify_downloads()
        
        logger.info("Script completed successfully!")
        logger.info(f"Final statistics: {stats}")
        
        # Print summary
        print(f"\n=== DOWNLOAD SUMMARY ===")
        print(f"Total legislation items: {stats['expected']}")
        print(f"XML files: {stats['xml_files']}")
        print(f"HTML files: {stats['html_files']}")
        print(f"Text files: {stats['text_files']}")
        print(f"Missing items: {stats['missing']}")
        print(f"Corrupted files: {stats['corrupted']}")
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        downloader.save_progress()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        downloader.save_progress()
        sys.exit(1)

if __name__ == "__main__":
    main()