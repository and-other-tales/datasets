#!/usr/bin/env python3
"""
ParaLlama Copyright Legislation Downloader

Specialised downloader for UK copyright and intellectual property legislation,
supporting Legal Llama's copyright law training datasets.
"""

import os
import re
import json
import time
import logging
import requests
from pathlib import Path
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CopyrightLegislationDownloader:
    def __init__(self, output_dir: str = "generated/copyright_legislation", max_items: Optional[int] = None):
        """
        Initialise ParaLlama Copyright Legislation Downloader
        
        Args:
            output_dir: Directory to store copyright legislation
            max_items: Maximum number of items to download
        """
        self.output_dir = Path(output_dir)
        self.max_items = max_items
        
        # Create output directories
        self.text_dir = self.output_dir / "text"
        self.html_dir = self.output_dir / "html"
        self.xml_dir = self.output_dir / "xml"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.text_dir, self.html_dir, self.xml_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ParaLlama-Copyright-Downloader/1.0 (Legal Research)'
        })
        
        # Copyright-specific legislation types and search terms
        self.copyright_legislation_types = [
            'ukpga',  # UK Public General Acts
            'uksi',   # UK Statutory Instruments  
            'eur',    # EU Legislation (historical)
            'eudr',   # EU Directives
        ]
        
        # Copyright and IP-related keywords
        self.copyright_keywords = [
            'copyright', 'intellectual property', 'patent', 'trademark', 'design',
            'moral rights', 'performers rights', 'database rights', 'broadcast rights',
            'authorship', 'originality', 'substantial part', 'fair dealing',
            'permitted acts', 'infringement', 'remedies', 'licensing',
            'collective licensing', 'orphan works', 'digital copyright',
            'CDPA', 'copyright designs and patents act', 'berne convention',
            'WIPO', 'trips agreement', 'information society directive',
            'rental and lending directive', 'software directive',
            'database directive', 'orphan works directive'
        ]
        
        # Tracking sets
        self.discovered_items = set()
        self.downloaded_items = set()
        self.failed_items = set()
        
        logger.info("ParaLlama Copyright Legislation Downloader initialised")
    
    def is_copyright_related(self, title: str, summary: str = "") -> bool:
        """Check if legislation is copyright/IP related"""
        text = (title + " " + summary).lower()
        
        # Direct copyright/IP terms
        for keyword in self.copyright_keywords:
            if keyword.lower() in text:
                return True
        
        # Additional checks for common IP terminology
        ip_patterns = [
            r'intellectual\s+property',
            r'copyright.*act',
            r'patent.*act', 
            r'trademark.*act',
            r'design.*act',
            r'performers.*rights',
            r'broadcast.*rights',
            r'database.*rights',
            r'moral\s+rights',
            r'fair\s+dealing',
            r'substantial\s+part'
        ]
        
        for pattern in ip_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def discover_copyright_legislation(self) -> Set[str]:
        """Discover copyright-related UK legislation"""
        logger.info("Discovering copyright and intellectual property legislation...")
        
        copyright_urls = set()
        base_url = "https://www.legislation.gov.uk"
        
        # Search by legislation type and keywords
        for leg_type in self.copyright_legislation_types:
            for keyword in ['copyright', 'intellectual property', 'patent', 'trademark', 'design']:
                try:
                    search_url = f"{base_url}/browse/{leg_type}?text={keyword}"
                    logger.info(f"Searching {leg_type} for '{keyword}'...")
                    
                    response = self.session.get(search_url, timeout=30)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find legislation links
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        title = link.get_text(strip=True)
                        
                        # Check if it's a legislation link
                        if (href.startswith(f'/{leg_type}/') and 
                            href.count('/') >= 3 and  # e.g., /ukpga/1988/48
                            self.is_copyright_related(title, href)):
                            
                            full_url = urljoin(base_url, href)
                            if full_url not in copyright_urls:
                                copyright_urls.add(full_url)
                                logger.info(f"Found copyright legislation: {title}")
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error searching {leg_type} for {keyword}: {e}")
                    continue
        
        # Add known key copyright legislation
        key_copyright_acts = [
            "https://www.legislation.gov.uk/ukpga/1988/48",  # Copyright, Designs and Patents Act 1988
            "https://www.legislation.gov.uk/uksi/2003/2498", # Copyright and Related Rights Regulations 2003
            "https://www.legislation.gov.uk/uksi/2014/2361", # Copyright and Rights in Performances Regulations 2014
            "https://www.legislation.gov.uk/uksi/2012/2989", # Orphan Works Licensing Regulations 2012
        ]
        
        for act_url in key_copyright_acts:
            copyright_urls.add(act_url)
        
        logger.info(f"Discovered {len(copyright_urls)} copyright-related legislation items")
        self.discovered_items = copyright_urls
        
        return copyright_urls
    
    def download_legislation_item(self, url: str) -> bool:
        """Download a single piece of copyright legislation"""
        try:
            # Parse URL to get identifier
            parsed = urlparse(url)
            path_parts = parsed.path.strip('/').split('/')
            
            if len(path_parts) < 3:
                logger.warning(f"Invalid legislation URL format: {url}")
                return False
            
            leg_type = path_parts[0]
            year = path_parts[1]
            number = path_parts[2]
            
            filename = f"{leg_type}_{year}_{number}"
            
            # Download main HTML page
            logger.info(f"Downloading {filename}...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title and summary
            title_elem = soup.find('h1') or soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else f"Unknown {filename}"
            
            # Extract main content
            content_selectors = [
                '.LegSnippet',
                '.LegContent', 
                '.ContentPrimary',
                'div.content',
                '#content'
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator='\n', strip=True)
                    break
            
            if not content:
                content = soup.get_text(separator='\n', strip=True)
            
            # Save HTML
            html_file = self.html_dir / f"{filename}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Save text content
            text_file = self.text_dir / f"{filename}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Try to download XML version
            xml_url = f"{url}/data.xml"
            try:
                xml_response = self.session.get(xml_url, timeout=30)
                if xml_response.status_code == 200:
                    xml_file = self.xml_dir / f"{filename}.xml"
                    with open(xml_file, 'w', encoding='utf-8') as f:
                        f.write(xml_response.text)
            except Exception:
                logger.debug(f"No XML version available for {filename}")
            
            # Create metadata
            metadata = {
                'url': url,
                'title': title,
                'legislation_type': leg_type,
                'year': year,
                'number': number,
                'filename': filename,
                'domain': 'copyright',
                'length': len(content),
                'downloaded_by': 'ParaLlama Copyright Downloader'
            }
            
            # Save metadata
            metadata_file = self.metadata_dir / f"{filename}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            self.downloaded_items.add(url)
            logger.info(f"Successfully downloaded: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            self.failed_items.add(url)
            return False
    
    def save_progress(self):
        """Save current progress"""
        progress = {
            'discovered_items': list(self.discovered_items),
            'downloaded_items': list(self.downloaded_items),
            'failed_items': list(self.failed_items),
            'total_discovered': len(self.discovered_items),
            'total_downloaded': len(self.downloaded_items),
            'total_failed': len(self.failed_items),
            'downloader': 'ParaLlama Copyright Legislation Downloader'
        }
        
        progress_file = self.output_dir / "progress.json"
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)
    
    def load_progress(self):
        """Load previous progress"""
        progress_file = self.output_dir / "progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                
                self.discovered_items = set(progress.get('discovered_items', []))
                self.downloaded_items = set(progress.get('downloaded_items', []))
                self.failed_items = set(progress.get('failed_items', []))
                
                logger.info(f"Loaded progress: {len(self.downloaded_items)} downloaded, {len(self.failed_items)} failed")
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
    
    def run_copyright_download(self) -> bool:
        """Run the complete copyright legislation download"""
        logger.info("=== STARTING PARALLAMA COPYRIGHT LEGISLATION DOWNLOAD ===")
        
        try:
            # Load existing progress
            self.load_progress()
            
            # Discover copyright legislation if not already discovered
            if not self.discovered_items:
                self.discover_copyright_legislation()
            
            # Apply limit if specified
            items_to_download = list(self.discovered_items - self.downloaded_items)
            if self.max_items:
                items_to_download = items_to_download[:self.max_items]
            
            logger.info(f"Downloading {len(items_to_download)} copyright legislation items...")
            
            # Download each item
            for i, url in enumerate(items_to_download, 1):
                logger.info(f"Progress: {i}/{len(items_to_download)}")
                
                success = self.download_legislation_item(url)
                
                if success:
                    logger.info(f"Downloaded {i}/{len(items_to_download)}")
                else:
                    logger.warning(f"Failed to download {i}/{len(items_to_download)}")
                
                # Rate limiting and progress saving
                time.sleep(2)
                
                if i % 10 == 0:
                    self.save_progress()
            
            # Final progress save
            self.save_progress()
            
            # Generate summary
            logger.info(f"=== PARALLAMA COPYRIGHT DOWNLOAD COMPLETE ===")
            logger.info(f"Total discovered: {len(self.discovered_items)}")
            logger.info(f"Successfully downloaded: {len(self.downloaded_items)}")
            logger.info(f"Failed downloads: {len(self.failed_items)}")
            
            return True
            
        except Exception as e:
            logger.error(f"ParaLlama copyright download failed: {e}")
            return False

def main():
    """Main function for ParaLlama Copyright Legislation Downloader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ParaLlama Copyright Legislation Downloader")
    parser.add_argument('--output-dir', default='generated/copyright_legislation',
                       help='Output directory for copyright legislation')
    parser.add_argument('--max-items', type=int,
                       help='Maximum number of items to download')
    
    args = parser.parse_args()
    
    downloader = CopyrightLegislationDownloader(args.output_dir, args.max_items)
    success = downloader.run_copyright_download()
    
    if success:
        print(f"\n🎉 ParaLlama copyright legislation download completed!")
        print(f"📂 Results saved to: {args.output_dir}")
    else:
        print(f"\n❌ ParaLlama copyright legislation download failed!")

if __name__ == "__main__":
    main()