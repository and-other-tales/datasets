#!/usr/bin/env python3
"""
HMRC Documentation Scraper

This script systematically downloads all tax-related documentation, forms, guidance, 
internal manuals for taxes, rebates and schemes from HMRC via gov.uk
"""

import os
import re
import json
import time
import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from collections import deque
import xml.etree.ElementTree as ET

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hmrc_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HMRCScraper:
    def __init__(self, output_dir: str = "hmrc_documentation"):
        self.base_url = "https://www.gov.uk"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different content types
        self.text_dir = self.output_dir / "text"
        self.html_dir = self.output_dir / "html"
        self.metadata_dir = self.output_dir / "metadata"
        self.forms_dir = self.output_dir / "forms"
        
        for dir_path in [self.text_dir, self.html_dir, self.metadata_dir, self.forms_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HMRC-Documentation-Scraper/1.0 (Educational/Research Purpose)'
        })
        
        # Tracking sets
        self.discovered_urls = set()
        self.downloaded_urls = set()
        self.failed_urls = set()
        
        # Tax-specific keywords for filtering
        self.tax_keywords = {
            'primary_terms': [
                'tax', 'taxes', 'taxation', 'vat', 'income tax', 'corporation tax',
                'capital gains', 'inheritance tax', 'stamp duty', 'national insurance',
                'paye', 'self assessment', 'hmrc', 'revenue', 'customs', 'duty',
                'allowance', 'relief', 'exemption', 'deduction', 'credit'
            ],
            'tax_types': [
                'income tax', 'corporation tax', 'capital gains tax', 'inheritance tax',
                'value added tax', 'vat', 'stamp duty land tax', 'annual tax on enveloped dwellings',
                'apprenticeship levy', 'bank levy', 'diverted profits tax',
                'petroleum revenue tax', 'landfill tax', 'climate change levy',
                'aggregates levy', 'air passenger duty', 'vehicle excise duty',
                'fuel duty', 'alcohol duty', 'tobacco duty', 'betting and gaming duties'
            ],
            'business_terms': [
                'self assessment', 'corporation tax return', 'vat return',
                'paye', 'payroll', 'benefits in kind', 'expenses',
                'business rates', 'annual investment allowance', 'research and development',
                'enterprise investment scheme', 'seed enterprise investment scheme',
                'venture capital trust', 'employee share schemes'
            ],
            'individual_terms': [
                'personal allowance', 'marriage allowance', 'blind person allowance',
                'tax credits', 'child benefit', 'working tax credit', 'child tax credit',
                'pension contributions', 'isa', 'premium bonds', 'savings',
                'dividends', 'interest', 'rental income', 'foreign income'
            ],
            'compliance_terms': [
                'penalty', 'appeal', 'enquiry', 'investigation', 'disclosure',
                'avoidance', 'evasion', 'compliance', 'record keeping',
                'registration', 'deregistration', 'making tax digital'
            ]
        }
        
        # Main search endpoints for HMRC content
        self.search_endpoints = [
            '/search/guidance-and-regulation?organisations%5B%5D=hm-revenue-customs',
            '/search/research-and-statistics?organisations%5B%5D=hm-revenue-customs',
            '/search/policy-papers-and-consultations?organisations%5B%5D=hm-revenue-customs',
            '/search/transparency?organisations%5B%5D=hm-revenue-customs',
            '/search/news-and-communications?organisations%5B%5D=hm-revenue-customs'
        ]
        
    def is_tax_related(self, title: str, summary: str = "") -> bool:
        """Check if content is tax-related"""
        text = (title + " " + summary).lower()
        
        # Check for specific tax-related terms
        all_tax_terms = []
        for category in self.tax_keywords.values():
            all_tax_terms.extend(category)
        
        tax_term_count = sum(1 for term in all_tax_terms if term in text)
        
        # Tax-related if it contains relevant terms
        return (
            tax_term_count >= 1 or
            any(term in text for term in ['tax', 'vat', 'hmrc', 'revenue', 'duty', 'allowance']) or
            'government/organisations/hm-revenue-customs' in text
        )
    
    def discover_guidance_documents(self, max_pages: int = 300) -> Set[str]:
        """Discover all guidance documents from HMRC"""
        logger.info("Discovering HMRC guidance documents...")
        
        guidance_urls = set()
        
        for endpoint in self.search_endpoints:
            logger.info(f"Searching endpoint: {endpoint}")
            page = 1
            
            while page <= max_pages:
                try:
                    # Build search URL with pagination
                    search_url = f"{self.base_url}{endpoint}"
                    if '?' in search_url:
                        search_url += f"&page={page}"
                    else:
                        search_url += f"?page={page}"
                    
                    response = self.session.get(search_url, timeout=30)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find document links
                    document_links = soup.find_all('a', href=True)
                    found_documents = 0
                    
                    for link in document_links:
                        href = link.get('href', '')
                        title = link.get_text(strip=True)
                        
                        # Check if it's a guidance document
                        if (href.startswith('/guidance/') or 
                            href.startswith('/government/publications/') or
                            href.startswith('/government/consultations/')):
                            
                            if self.is_tax_related(title, href):
                                full_url = urljoin(self.base_url, href)
                                if full_url not in guidance_urls:
                                    guidance_urls.add(full_url)
                                    found_documents += 1
                                    logger.info(f"Found: {title}")
                    
                    if found_documents == 0:
                        logger.info(f"No more documents found on page {page} for {endpoint}")
                        break
                    
                    logger.info(f"Page {page}: Found {found_documents} documents")
                    page += 1
                    time.sleep(0.1)  # Rate limiting for Content API (10 req/sec)
                    
                except requests.RequestException as e:
                    logger.error(f"Error fetching page {page} from {endpoint}: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error on page {page}: {e}")
                    break
        
        logger.info(f"Discovered {len(guidance_urls)} total guidance documents")
        return guidance_urls
    
    def discover_forms(self) -> Set[str]:
        """Discover tax forms and documents"""
        logger.info("Discovering HMRC forms...")
        
        forms_search_terms = [
            'form', 'return', 'declaration', 'application', 'claim',
            'sa100', 'sa200', 'sa800', 'ct600', 'vat100',
            'p45', 'p46', 'p60', 'p11d', 'r40'
        ]
        
        form_urls = set()
        
        for search_term in forms_search_terms:
            try:
                search_url = f"{self.base_url}/search/all?keywords={search_term}&organisations%5B%5D=hm-revenue-customs"
                response = self.session.get(search_url, timeout=30)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        title = link.get_text(strip=True)
                        
                        if (href.startswith('/government/publications/') and
                            ('form' in title.lower() or 'return' in title.lower() or
                             any(term in title.lower() for term in forms_search_terms))):
                            
                            full_url = urljoin(self.base_url, href)
                            form_urls.add(full_url)
                            logger.info(f"Found form: {title}")
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Error searching for forms with term '{search_term}': {e}")
        
        logger.info(f"Discovered {len(form_urls)} forms")
        return form_urls
    
    def discover_manuals(self) -> Set[str]:
        """Discover internal manuals and detailed guidance"""
        logger.info("Discovering HMRC manuals...")
        
        manual_search_terms = [
            'manual', 'handbook', 'guidance', 'instructions',
            'technical', 'procedural', 'operational'
        ]
        
        manual_urls = set()
        
        # Search for manuals
        for search_term in manual_search_terms:
            try:
                search_url = f"{self.base_url}/search/guidance-and-regulation?keywords={search_term}&organisations%5B%5D=hm-revenue-customs"
                response = self.session.get(search_url, timeout=30)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        title = link.get_text(strip=True)
                        
                        if href.startswith('/guidance/') and self.is_tax_related(title):
                            full_url = urljoin(self.base_url, href)
                            manual_urls.add(full_url)
                            logger.info(f"Found manual: {title}")
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Error searching for manuals with term '{search_term}': {e}")
        
        logger.info(f"Discovered {len(manual_urls)} manuals")
        return manual_urls
    
    def get_api_url(self, web_url: str) -> str:
        """Convert web URL to Content API URL"""
        try:
            parsed = urlparse(web_url)
            path = parsed.path
            if path.startswith('/'):
                path = path[1:]
            return f"{self.base_url}/api/content/{path}"
        except Exception:
            return None

    def extract_content_from_api(self, api_url: str) -> Optional[Dict]:
        """Extract content using the GOV.UK Content API"""
        try:
            response = self.session.get(api_url, timeout=30)
            response.raise_for_status()
            
            api_data = response.json()
            
            # Extract structured content
            title = api_data.get('title', 'Unknown Title')
            description = api_data.get('description', '')
            
            # Extract main content from details
            details = api_data.get('details', {})
            content_parts = []
            
            # Common content fields in GOV.UK API
            for field in ['body', 'parts', 'introduction', 'more_information']:
                if field in details:
                    if isinstance(details[field], str):
                        content_parts.append(details[field])
                    elif isinstance(details[field], list):
                        for part in details[field]:
                            if isinstance(part, dict) and 'body' in part:
                                content_parts.append(part['body'])
                            elif isinstance(part, str):
                                content_parts.append(part)
            
            # If no structured content found, use the entire details as text
            if not content_parts:
                content_parts.append(str(details))
            
            content = '\n\n'.join(content_parts)
            
            # Clean HTML tags from content if present
            if '<' in content and '>' in content:
                soup = BeautifulSoup(content, 'html.parser')
                content = soup.get_text(separator='\n', strip=True)
            
            # Extract metadata from API response
            metadata = {
                'url': api_data.get('base_path', ''),
                'content_id': api_data.get('content_id', ''),
                'title': title,
                'description': description,
                'last_updated': api_data.get('updated_at') or api_data.get('public_updated_at'),
                'first_published': api_data.get('first_published_at'),
                'organisation': 'HM Revenue and Customs',
                'content_type': api_data.get('document_type', 'unknown'),
                'schema_name': api_data.get('schema_name', ''),
                'length': len(content),
                'api_source': True
            }
            
            # Extract links to related content
            links = api_data.get('links', {})
            if links:
                metadata['related_links'] = links
            
            return {
                'metadata': metadata,
                'content': content,
                'api_data': api_data
            }
            
        except requests.RequestException as e:
            logger.debug(f"API request failed for {api_url}: {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.debug(f"API response parsing failed for {api_url}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error with API for {api_url}: {e}")
            return None

    def extract_content_from_html(self, url: str) -> Optional[Dict]:
        """Extract content using traditional HTML scraping (fallback method)"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1') or soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else "Unknown Title"
            
            # Extract main content
            content_selectors = [
                '.gem-c-govspeak',
                '.govuk-govspeak',
                '#content',
                '.publication-external-link',
                'main',
                '.govuk-main-wrapper'
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator='\n', strip=True)
                    break
            
            if not content:
                content = soup.get_text(separator='\n', strip=True)
            
            # Extract metadata
            metadata = {
                'url': url,
                'title': title,
                'last_updated': None,
                'organisation': 'HM Revenue and Customs',
                'content_type': 'guidance',
                'length': len(content),
                'api_source': False
            }
            
            # Try to extract last updated date
            date_elem = soup.find('time') or soup.find(class_='gem-c-metadata__definition')
            if date_elem:
                metadata['last_updated'] = date_elem.get_text(strip=True)
            
            # Determine content type
            if '/guidance/' in url:
                metadata['content_type'] = 'guidance'
            elif '/government/publications/' in url:
                metadata['content_type'] = 'publication'
            elif '/government/consultations/' in url:
                metadata['content_type'] = 'consultation'
            elif 'form' in title.lower():
                metadata['content_type'] = 'form'
            
            return {
                'metadata': metadata,
                'content': content,
                'html': response.text
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None

    def extract_document_content(self, url: str) -> Optional[Dict]:
        """Extract content from a single document using Content API with HTML fallback"""
        # Try Content API first
        api_url = self.get_api_url(url)
        if api_url:
            logger.debug(f"Attempting Content API extraction for {url}")
            api_result = self.extract_content_from_api(api_url)
            if api_result:
                logger.debug(f"Successfully extracted via Content API: {url}")
                return api_result
        
        # Fallback to HTML scraping
        logger.debug(f"Falling back to HTML extraction for {url}")
        html_result = self.extract_content_from_html(url)
        if html_result:
            logger.debug(f"Successfully extracted via HTML scraping: {url}")
        
        return html_result
    
    def download_document(self, url: str) -> bool:
        """Download a single document"""
        try:
            # Generate filename from URL
            url_path = urlparse(url).path
            filename = re.sub(r'[^\w\-_.]', '_', url_path.split('/')[-1])
            if not filename or filename == '_':
                filename = re.sub(r'[^\w\-_.]', '_', url_path)
            
            # Extract document content
            doc_data = self.extract_document_content(url)
            if not doc_data:
                return False
            
            # Save text content
            text_file = self.text_dir / f"{filename}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(doc_data['content'])
            
            # Save HTML content (if available) or API data
            if 'html' in doc_data:
                html_file = self.html_dir / f"{filename}.html"
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(doc_data['html'])
            elif 'api_data' in doc_data:
                api_file = self.html_dir / f"{filename}_api.json"
                with open(api_file, 'w', encoding='utf-8') as f:
                    json.dump(doc_data['api_data'], f, indent=2)
            
            # Save metadata
            metadata_file = self.metadata_dir / f"{filename}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(doc_data['metadata'], f, indent=2)
            
            self.downloaded_urls.add(url)
            logger.info(f"Downloaded: {doc_data['metadata']['title']}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            self.failed_urls.add(url)
            return False
    
    def run_comprehensive_discovery(self):
        """Run comprehensive discovery of all HMRC documentation"""
        logger.info("=== STARTING COMPREHENSIVE HMRC DISCOVERY ===")
        
        # Load existing progress
        self.load_progress()
        
        # Discover all types of documents
        guidance_urls = self.discover_guidance_documents()
        form_urls = self.discover_forms()
        manual_urls = self.discover_manuals()
        
        # Combine all URLs
        all_urls = guidance_urls.union(form_urls).union(manual_urls)
        self.discovered_urls = all_urls
        
        logger.info(f"Total unique documents discovered: {len(all_urls)}")
        logger.info(f"Guidance documents: {len(guidance_urls)}")
        logger.info(f"Forms: {len(form_urls)}")
        logger.info(f"Manuals: {len(manual_urls)}")
        
        # Save discovery results
        self.save_progress()
        
        # Save discovered URLs for reference
        with open(self.output_dir / "discovered_urls.json", 'w') as f:
            json.dump({
                'guidance': list(guidance_urls),
                'forms': list(form_urls),
                'manuals': list(manual_urls),
                'total': len(all_urls)
            }, f, indent=2)
    
    def download_all_documents(self, max_documents: Optional[int] = None):
        """Download all discovered documents"""
        if not self.discovered_urls:
            logger.warning("No documents discovered. Running discovery first...")
            self.run_comprehensive_discovery()
        
        urls_to_download = list(self.discovered_urls - self.downloaded_urls)
        
        if max_documents:
            urls_to_download = urls_to_download[:max_documents]
        
        logger.info(f"Starting download of {len(urls_to_download)} documents...")
        
        for i, url in enumerate(urls_to_download, 1):
            logger.info(f"Progress: {i}/{len(urls_to_download)}")
            
            success = self.download_document(url)
            
            if success:
                logger.info(f"Successfully downloaded {i}/{len(urls_to_download)}")
            else:
                logger.warning(f"Failed to download {i}/{len(urls_to_download)}")
            
            # Rate limiting for Content API (10 req/sec)
            time.sleep(0.1)
            
            # Save progress periodically
            if i % 50 == 0:
                self.save_progress()
        
        self.save_progress()
        logger.info(f"Download complete. Success: {len(self.downloaded_urls)}, Failed: {len(self.failed_urls)}")
    
    def save_progress(self):
        """Save current progress"""
        progress = {
            'discovered_urls': list(self.discovered_urls),
            'downloaded_urls': list(self.downloaded_urls),
            'failed_urls': list(self.failed_urls),
            'total_discovered': len(self.discovered_urls),
            'total_downloaded': len(self.downloaded_urls),
            'total_failed': len(self.failed_urls)
        }
        
        with open(self.output_dir / "progress.json", 'w') as f:
            json.dump(progress, f, indent=2)
    
    def load_progress(self):
        """Load previous progress"""
        progress_file = self.output_dir / "progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            
            self.discovered_urls = set(progress.get('discovered_urls', []))
            self.downloaded_urls = set(progress.get('downloaded_urls', []))
            self.failed_urls = set(progress.get('failed_urls', []))
            
            logger.info(f"Loaded progress: {len(self.downloaded_urls)} downloaded, {len(self.failed_urls)} failed")
    
    def generate_summary(self):
        """Generate summary of downloaded HMRC documentation"""
        summary = {
            'total_discovered': len(self.discovered_urls),
            'total_downloaded': len(self.downloaded_urls),
            'total_failed': len(self.failed_urls),
            'content_types': {},
            'tax_categories': {},
            'file_stats': {}
        }
        
        # Analyze downloaded content
        for metadata_file in self.metadata_dir.glob('*.json'):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            content_type = metadata.get('content_type', 'unknown')
            summary['content_types'][content_type] = summary['content_types'].get(content_type, 0) + 1
        
        # Count files
        summary['file_stats'] = {
            'text_files': len(list(self.text_dir.glob('*.txt'))),
            'html_files': len(list(self.html_dir.glob('*.html'))),
            'metadata_files': len(list(self.metadata_dir.glob('*.json')))
        }
        
        # Save summary
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    """Main function to run the HMRC scraper"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape HMRC tax documentation")
    parser.add_argument('--output-dir', default='hmrc_documentation',
                       help='Directory to store HMRC documentation')
    parser.add_argument('--max-documents', type=int,
                       help='Maximum number of documents to download')
    parser.add_argument('--discover-only', action='store_true',
                       help='Only discover URLs, do not download content')
    
    args = parser.parse_args()
    
    scraper = HMRCScraper(args.output_dir)
    
    try:
        if args.discover_only:
            scraper.run_comprehensive_discovery()
            print(f"Discovered {len(scraper.discovered_urls)} HMRC documents")
        else:
            scraper.run_comprehensive_discovery()
            scraper.download_all_documents(args.max_documents)
            summary = scraper.generate_summary()
            
            print(f"\n=== HMRC DOCUMENTATION SCRAPING COMPLETE ===")
            print(f"Total documents discovered: {summary['total_discovered']}")
            print(f"Total documents downloaded: {summary['total_downloaded']}")
            print(f"Total failed downloads: {summary['total_failed']}")
            print(f"Content types: {summary['content_types']}")
            print(f"Output directory: {args.output_dir}")
            
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        scraper.save_progress()
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        scraper.save_progress()

if __name__ == "__main__":
    main()