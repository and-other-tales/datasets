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

# Import HMRC metadata framework
try:
    from utils.hmrc_metadata import HMRCDocumentProcessor, HMRCMetadata, TaxDomain, save_hmrc_metadata
except ImportError:
    # Fallback if not in package context
    import sys
    sys.path.append(str(Path(__file__).parent / 'utils'))
    from hmrc_metadata import HMRCDocumentProcessor, HMRCMetadata, TaxDomain, save_hmrc_metadata

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    logger = logging.getLogger(__name__)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    # Always add file handler
    file_handler = logging.FileHandler('hmrc_scraper.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Only add StreamHandler if not running under curses
    # Curses environments will add their own handlers
    if not any('curses' in str(handler) for handler in logging.root.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(stream_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

class HMRCScraper:
    def __init__(self, output_dir: str = "hmrc_documentation"):
        self.base_url = "https://www.gov.uk"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different content types
        self.text_dir = self.output_dir / "text"
        self.html_dir = self.output_dir / "html"
        self.metadata_dir = self.output_dir / "metadata"
        self.enhanced_metadata_dir = self.output_dir / "enhanced_metadata"
        self.forms_dir = self.output_dir / "forms"
        
        # Create tax category subdirectories
        self.category_dirs = {}
        for domain in TaxDomain:
            category_dir = self.output_dir / "categorized" / domain.value
            category_dir.mkdir(parents=True, exist_ok=True)
            self.category_dirs[domain] = category_dir
        
        for dir_path in [self.text_dir, self.html_dir, self.metadata_dir, 
                        self.enhanced_metadata_dir, self.forms_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize HMRC document processor
        self.hmrc_processor = HMRCDocumentProcessor()
        
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
        
        # GOV.UK Search API configuration
        self.search_api_base = f"{self.base_url}/api/search.json"
        self.batch_size = 100  # Reduced for reliability
        
        # Priority content formats for tax advice
        self.priority_formats = [
            'guide', 'detailed_guide', 'manual', 'answer',
            'form', 'publication', 'consultation_outcome'
        ]
        
        # Content quality filters
        self.quality_filters = {
            'exclude_formats': ['press_release', 'news_story', 'speech'],
            'min_content_length': 200,
            'require_recent_update': False  # Set to True for only recent content
        }
        
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
    
    def is_high_quality_tax_content(self, title: str, description: str, format_type: str, result: dict) -> bool:
        """Enhanced quality assessment for tax advice content"""
        # Log what we're checking
        logger.debug(f"Checking quality for: {title} (format: {format_type})")
        
        # Must be tax-related first
        if not self.is_tax_related(title, description):
            logger.debug(f"Not tax-related: {title}")
            return False
        
        # Prioritize high-value formats
        if format_type in self.priority_formats:
            score = 10
        else:
            score = 1
        
        # Exclude low-value formats
        if format_type in self.quality_filters['exclude_formats']:
            logger.debug(f"Excluded format {format_type}: {title}")
            return False
        
        # Boost score for actionable tax advice indicators
        actionable_terms = [
            'how to', 'calculate', 'claim', 'apply', 'rate', 'allowance',
            'deadline', 'form', 'return', 'guidance', 'rules', 'requirements'
        ]
        text = (title + " " + description).lower()
        score += sum(2 for term in actionable_terms if term in text)
        
        # Boost for specific tax areas
        specific_areas = [
            'self assessment', 'corporation tax', 'vat', 'paye', 'capital gains',
            'inheritance tax', 'stamp duty', 'making tax digital', 'tax credits'
        ]
        score += sum(3 for area in specific_areas if area in text)
        
        # Check minimum content quality threshold - reduced from 5 to 3
        passed = score >= 3
        logger.debug(f"Quality score for '{title}': {score} (passed: {passed})")
        return passed
    
    def discover_via_search_api(self) -> Set[str]:
        """Discover all HMRC documents using the GOV.UK Search API"""
        logger.info("Discovering HMRC documents via Search API...")
        
        all_urls = set()
        start = 0
        
        while True:
            try:
                # Build API query with pagination
                params = {
                    'count': self.batch_size,
                    'start': start,
                    'filter_organisations': 'hm-revenue-customs',
                    'order': '-public_timestamp'  # Most recent first
                }
                
                # Note: The GOV.UK API doesn't support filter_any_format or reject_format
                # We'll filter results manually after retrieval
                
                logger.info(f"Making API request to {self.search_api_base} with params: {params}")
                response = self.session.get(self.search_api_base, params=params, timeout=30)
                response.raise_for_status()
                
                # Log response status and headers
                logger.info(f"API Response status: {response.status_code}")
                logger.debug(f"API Response headers: {dict(response.headers)}")
                
                # Ensure we wait for complete response
                response_text = response.text
                if not response_text:
                    logger.error("Empty response from API")
                    break
                    
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Response text: {response_text[:500]}")
                    break
                
                results = data.get('results', [])
                total_results = data.get('total', 0)
                
                logger.info(f"API returned {len(results)} results out of {total_results} total")
                
                if not results:
                    logger.info(f"No more results found at start={start}")
                    break
                
                batch_urls = set()
                all_results_count = 0
                for result in results:
                    all_results_count += 1
                    title = result.get('title', '')
                    description = result.get('description', '')
                    link = result.get('link', '')
                    format_type = result.get('format', '')
                    
                    # Log first few results to debug
                    if all_results_count <= 3:
                        logger.info(f"Sample result: title='{title}', format='{format_type}', link='{link}'")
                    
                    # Apply quality filters
                    if self.is_high_quality_tax_content(title, description, format_type, result):
                        full_url = urljoin(self.base_url, link)
                        batch_urls.add(full_url)
                        all_urls.add(full_url)
                        logger.debug(f"Found: {title} ({format_type})")
                
                # Log batch info with current/total context
                batch_end = start + len(results)
                progress_pct = (batch_end / total_results * 100) if total_results > 0 else 0
                logger.info(f"Batch {start:06d}-{batch_end:06d}: Found {len(batch_urls)} relevant documents out of {len(results)} total ({progress_pct:.1f}% complete)")
                start += len(results)
                
                # API rate limiting
                time.sleep(0.1)
                    
            except requests.RequestException as e:
                logger.error(f"Error fetching batch starting at {start}: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error at batch {start}: {e}")
                break
        
        logger.info(f"Discovered {len(all_urls)} total relevant documents")
        
        # If no results found, try simpler search without organization filter
        if len(all_urls) == 0:
            logger.warning("No results found with HMRC filter. Trying alternative search...")
            all_urls = self.discover_via_search_alternative()
        
        return all_urls
    
    def discover_via_search_alternative(self) -> Set[str]:
        """Alternative discovery method using search terms"""
        logger.info("Using alternative search method...")
        
        all_urls = set()
        
        # Search for HMRC-specific content using keywords
        search_terms = ['HMRC tax', 'income tax uk', 'vat return', 'self assessment', 'corporation tax uk']
        
        for search_term in search_terms:
            try:
                params = {
                    'q': search_term,
                    'count': 50,
                    'start': 0
                }
                
                logger.info(f"Searching for: {search_term}")
                response = self.session.get(self.search_api_base, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                results = data.get('results', [])
                
                for result in results:
                    link = result.get('link', '')
                    title = result.get('title', '')
                    
                    # Filter for HMRC content
                    if link and ('hmrc' in link.lower() or 'hm-revenue-customs' in link or 
                                'tax' in title.lower() or 'vat' in title.lower()):
                        full_url = urljoin(self.base_url, link)
                        all_urls.add(full_url)
                        logger.info(f"Found via search: {title}")
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in alternative search for '{search_term}': {e}")
        
        logger.info(f"Alternative search discovered {len(all_urls)} documents")
        return all_urls
    
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
            return ''

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
        """Download a single document with enhanced tax categorization"""
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
            
            # Process with HMRC metadata framework
            enhanced_metadata = self.hmrc_processor.process_hmrc_document(
                text=doc_data['content'],
                title=doc_data['metadata'].get('title', ''),
                url=url,
                manual_code=doc_data['metadata'].get('manual_code', '')
            )
            
            # Determine tax category
            tax_category = enhanced_metadata.tax_domain
            category_dir = self.category_dirs[tax_category]
            
            # Save text content (both main directory and categorized)
            text_file = self.text_dir / f"{filename}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(doc_data['content'])
            
            # Save categorized text content
            categorized_text_file = category_dir / f"{filename}.txt"
            with open(categorized_text_file, 'w', encoding='utf-8') as f:
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
            
            # Save basic metadata
            metadata_file = self.metadata_dir / f"{filename}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(doc_data['metadata'], f, indent=2)
            
            # Save enhanced metadata
            enhanced_metadata_file = self.enhanced_metadata_dir / f"{filename}.json"
            save_hmrc_metadata(enhanced_metadata, enhanced_metadata_file)
            
            # Save enhanced metadata in category directory
            categorized_metadata_file = category_dir / f"{filename}_metadata.json"
            save_hmrc_metadata(enhanced_metadata, categorized_metadata_file)
            
            self.downloaded_urls.add(url)
            logger.info(f"Downloaded and categorized: {enhanced_metadata.title} -> {tax_category.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            self.failed_urls.add(url)
            return False
    
    def run_comprehensive_discovery(self):
        """Run comprehensive discovery of all HMRC documentation using Search API"""
        logger.info("=== STARTING COMPREHENSIVE HMRC DISCOVERY VIA SEARCH API ===")
        
        # Load existing progress
        self.load_progress()
        
        # Use the new Search API method
        all_urls = self.discover_via_search_api()
        self.discovered_urls = all_urls
        
        logger.info(f"Total high-quality tax documents discovered: {len(all_urls)}")
        
        # Save discovery results
        self.save_progress()
        
        # Save discovered URLs for reference
        with open(self.output_dir / "discovered_urls.json", 'w') as f:
            json.dump({
                'api_discovered': list(all_urls),
                'total': len(all_urls),
                'discovery_method': 'search_api',
                'filters_applied': {
                    'priority_formats': self.priority_formats,
                    'quality_filters': self.quality_filters
                }
            }, f, indent=2)
    
    def download_all_documents(self, max_documents: Optional[int] = None):
        """Download all discovered documents with resume capability"""
        if not self.discovered_urls:
            logger.warning("No documents discovered. Running discovery first...")
            self.run_comprehensive_discovery()
        
        # Filter out already downloaded documents
        urls_to_download = list(self.discovered_urls - self.downloaded_urls)
        
        # Verify existing downloads to catch incomplete ones
        logger.info("Verifying integrity of existing downloads...")
        incomplete_downloads = []
        verified_count = 0
        
        for url in list(self.downloaded_urls):
            if not self.verify_download_integrity(url):
                incomplete_downloads.append(url)
                self.downloaded_urls.discard(url)  # Remove from downloaded set
                logger.warning(f"Incomplete download detected, will retry: {url}")
            else:
                verified_count += 1
        
        logger.info(f"Verified {verified_count} complete downloads, found {len(incomplete_downloads)} incomplete")
        
        # Add incomplete downloads back to the download queue
        urls_to_download.extend(incomplete_downloads)
        urls_to_download = list(set(urls_to_download))  # Remove duplicates
        
        if max_documents:
            # If resuming, consider already downloaded count
            remaining_to_download = max_documents - len(self.downloaded_urls)
            if remaining_to_download <= 0:
                logger.info(f"Already downloaded {len(self.downloaded_urls)} documents, max_documents ({max_documents}) reached")
                return
            urls_to_download = urls_to_download[:remaining_to_download]
        
        if not urls_to_download:
            logger.info("No new documents to download. All discovered documents already downloaded.")
            return
        
        total_target = len(self.downloaded_urls) + len(urls_to_download)
        logger.info(f"Resuming download: {len(self.downloaded_urls)} already downloaded, {len(urls_to_download)} remaining")
        logger.info(f"Target total: {total_target} documents")
        
        try:
            for i, url in enumerate(urls_to_download, 1):
                current_total = len(self.downloaded_urls) + i
                logger.info(f"Progress: {current_total}/{total_target} (downloading {i}/{len(urls_to_download)}) - {url}")
                
                try:
                    success = self.download_document(url)
                    
                    if success:
                        logger.info(f"Successfully downloaded {current_total}/{total_target}")
                    else:
                        logger.warning(f"Failed to download {current_total}/{total_target}")
                        
                except Exception as e:
                    logger.error(f"Error downloading document {i}: {e}")
                    self.failed_urls.add(url)
                
                # Rate limiting for Content API (10 req/sec)
                time.sleep(0.1)
                
                # Save progress more frequently for large downloads
                if i % 25 == 0:  # Save every 25 documents instead of 50
                    logger.info(f"Saving progress at document {current_total}/{total_target}")
                    self.save_progress()
            
            # Final save and summary
            self.save_progress()
            logger.info(f"Download phase complete. Total downloaded: {len(self.downloaded_urls)}, Failed: {len(self.failed_urls)}")
            
        except Exception as e:
            logger.error(f"Critical error during download process: {e}")
            self.save_progress()
            raise
    
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
        """Load previous progress and scan existing files"""
        progress_file = self.output_dir / "progress.json"
        
        # Load from progress file if it exists
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            
            self.discovered_urls = set(progress.get('discovered_urls', []))
            self.downloaded_urls = set(progress.get('downloaded_urls', []))
            self.failed_urls = set(progress.get('failed_urls', []))
            
            logger.info(f"Loaded progress from file: {len(self.downloaded_urls)} downloaded, {len(self.failed_urls)} failed")
        
        # Scan existing files to detect what's already been downloaded
        existing_downloads = self.scan_existing_downloads()
        
        # Merge existing downloads with loaded progress
        if existing_downloads:
            original_count = len(self.downloaded_urls)
            self.downloaded_urls.update(existing_downloads)
            new_count = len(self.downloaded_urls) - original_count
            
            if new_count > 0:
                logger.info(f"Detected {new_count} additional completed downloads from filesystem scan")
            
        logger.info(f"Total progress: {len(self.downloaded_urls)} downloaded, {len(self.failed_urls)} failed")
    
    def scan_existing_downloads(self) -> Set[str]:
        """Scan filesystem to detect already downloaded documents"""
        existing_urls = set()
        
        # Check metadata files to reconstruct URLs
        if self.metadata_dir.exists():
            for metadata_file in self.metadata_dir.glob('*.json'):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Get URL from metadata
                    url = metadata.get('url', '')
                    if url:
                        # Ensure it's a full URL
                        if not url.startswith('http'):
                            url = urljoin(self.base_url, url)
                        existing_urls.add(url)
                        
                except Exception as e:
                    logger.debug(f"Error reading metadata {metadata_file}: {e}")
        
        # Also check enhanced metadata
        if self.enhanced_metadata_dir.exists():
            for metadata_file in self.enhanced_metadata_dir.glob('*.json'):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    url = metadata.get('source_url', '')
                    if url:
                        if not url.startswith('http'):
                            url = urljoin(self.base_url, url)
                        existing_urls.add(url)
                        
                except Exception as e:
                    logger.debug(f"Error reading enhanced metadata {metadata_file}: {e}")
        
        if existing_urls:
            logger.info(f"Filesystem scan found {len(existing_urls)} existing downloads")
        
        return existing_urls
    
    def verify_download_integrity(self, url: str) -> bool:
        """Verify that a download is complete and valid"""
        try:
            # Generate filename from URL
            url_path = urlparse(url).path
            filename = re.sub(r'[^\w\-_.]', '_', url_path.split('/')[-1])
            if not filename or filename == '_':
                filename = re.sub(r'[^\w\-_.]', '_', url_path)
            
            # Check if all required files exist
            text_file = self.text_dir / f"{filename}.txt"
            metadata_file = self.metadata_dir / f"{filename}.json"
            enhanced_metadata_file = self.enhanced_metadata_dir / f"{filename}.json"
            
            # All three files should exist for a complete download
            files_exist = text_file.exists() and metadata_file.exists() and enhanced_metadata_file.exists()
            
            if files_exist:
                # Check if files have content
                text_size = text_file.stat().st_size if text_file.exists() else 0
                metadata_size = metadata_file.stat().st_size if metadata_file.exists() else 0
                
                # Files should have meaningful content
                return text_size > 50 and metadata_size > 50
            
            return False
            
        except Exception as e:
            logger.debug(f"Error verifying download integrity for {url}: {e}")
            return False
    
    def generate_summary(self):
        """Generate summary of downloaded HMRC documentation with tax categorization"""
        summary = {
            'total_discovered': len(self.discovered_urls),
            'total_downloaded': len(self.downloaded_urls),
            'total_failed': len(self.failed_urls),
            'content_types': {},
            'tax_categories': {},
            'authority_levels': {},
            'entity_coverage': {
                'individuals': 0,
                'companies': 0,
                'trusts': 0,
                'partnerships': 0
            },
            'file_stats': {}
        }
        
        # Analyze enhanced metadata
        for metadata_file in self.enhanced_metadata_dir.glob('*.json'):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Count by tax domain
                tax_domain = metadata.get('tax_domain', 'unknown')
                summary['tax_categories'][tax_domain] = summary['tax_categories'].get(tax_domain, 0) + 1
                
                # Count by document type
                doc_type = metadata.get('document_type', 'unknown')
                summary['content_types'][doc_type] = summary['content_types'].get(doc_type, 0) + 1
                
                # Count by authority level
                authority = metadata.get('authority_level', 'unknown')
                summary['authority_levels'][authority] = summary['authority_levels'].get(authority, 0) + 1
                
                # Count entity coverage
                if metadata.get('affects_individuals', False):
                    summary['entity_coverage']['individuals'] += 1
                if metadata.get('affects_companies', False):
                    summary['entity_coverage']['companies'] += 1
                if metadata.get('affects_trusts', False):
                    summary['entity_coverage']['trusts'] += 1
                if metadata.get('affects_partnerships', False):
                    summary['entity_coverage']['partnerships'] += 1
                    
            except Exception as e:
                logger.warning(f"Error reading enhanced metadata {metadata_file}: {e}")
        
        # Count files by category
        category_stats = {}
        for domain, category_dir in self.category_dirs.items():
            txt_count = len(list(category_dir.glob('*.txt')))
            meta_count = len(list(category_dir.glob('*_metadata.json')))
            category_stats[domain.value] = {
                'documents': txt_count,
                'metadata': meta_count
            }
        
        summary['file_stats'] = {
            'text_files': len(list(self.text_dir.glob('*.txt'))),
            'html_files': len(list(self.html_dir.glob('*.html'))),
            'basic_metadata_files': len(list(self.metadata_dir.glob('*.json'))),
            'enhanced_metadata_files': len(list(self.enhanced_metadata_dir.glob('*.json'))),
            'category_breakdown': category_stats
        }
        
        # Save summary
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def create_training_datasets(self):
        """Create training datasets from downloaded HMRC content"""
        try:
            from utils.dataset_creator import DatasetCreator
            
            logger.info("Creating training datasets from HMRC content...")
            
            # Initialize dataset creator
            dataset_creator = DatasetCreator(
                input_dir=str(self.output_dir),
                output_dir=str(self.output_dir / "datasets")
            )
            
            # Create datasets by tax category
            for domain, category_dir in self.category_dirs.items():
                if len(list(category_dir.glob('*.txt'))) > 0:
                    logger.info(f"Creating dataset for {domain.value}...")
                    
                    dataset_creator.create_tax_category_dataset(
                        category=domain.value,
                        category_dir=str(category_dir),
                        min_documents=5  # Only create if we have at least 5 documents
                    )
            
            # Create comprehensive dataset
            logger.info("Creating comprehensive HMRC dataset...")
            dataset_creator.create_comprehensive_hmrc_dataset(
                enhanced_metadata_dir=str(self.enhanced_metadata_dir),
                text_dir=str(self.text_dir)
            )
            
            logger.info("Training dataset creation complete!")
            
        except ImportError:
            logger.warning("DatasetCreator not available. Skipping dataset creation.")
        except Exception as e:
            logger.error(f"Error creating training datasets: {e}")

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
    parser.add_argument('--resume', action='store_true',
                       help='Resume previous download (scan existing files and continue)')
    parser.add_argument('--verify-integrity', action='store_true',
                       help='Verify integrity of existing downloads and re-download incomplete ones')
    
    args = parser.parse_args()
    
    scraper = HMRCScraper(args.output_dir)
    
    try:
        if args.discover_only:
            scraper.run_comprehensive_discovery()
            print(f"Discovered {len(scraper.discovered_urls)} HMRC documents")
        elif args.verify_integrity:
            print("=== VERIFYING DOWNLOAD INTEGRITY ===")
            scraper.load_progress()  # This will scan existing files
            
            # Check all downloaded URLs for integrity
            incomplete_urls = []
            for url in list(scraper.downloaded_urls):
                if not scraper.verify_download_integrity(url):
                    incomplete_urls.append(url)
                    
            print(f"Found {len(incomplete_urls)} incomplete downloads")
            if incomplete_urls:
                for url in incomplete_urls:
                    print(f"  - {url}")
        else:
            if args.resume:
                print("=== RESUMING PREVIOUS DOWNLOAD ===")
            else:
                print("=== STARTING FRESH DOWNLOAD ===")
                
            scraper.run_comprehensive_discovery()
            scraper.download_all_documents(args.max_documents)
            summary = scraper.generate_summary()
            
            print(f"\n=== HMRC DOCUMENTATION SCRAPING COMPLETE ===")
            print(f"Total documents discovered: {summary['total_discovered']}")
            print(f"Total documents downloaded: {summary['total_downloaded']}")
            print(f"Total failed downloads: {summary['total_failed']}")
            print(f"Content types: {summary['content_types']}")
            print(f"Tax categories: {summary['tax_categories']}")
            print(f"Output directory: {args.output_dir}")
            
            # Auto-create datasets from downloaded content
            if summary['total_downloaded'] > 0:
                print(f"\n=== CREATING DATASETS FROM DOWNLOADED CONTENT ===")
                scraper.create_training_datasets()
                print("Dataset creation complete!")
            
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        scraper.save_progress()
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        scraper.save_progress()

if __name__ == "__main__":
    main()