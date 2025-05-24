import requests
from bs4 import BeautifulSoup
import json
import time
from tqdm import tqdm
import re
import textwrap
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging
from anthropic import Anthropic
from urllib.parse import urljoin, urlparse
from collections import deque

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.pipeline_controller import PipelineController, create_database_update_callback, create_dataset_creation_callback
from utils.rate_limiter import RateLimiter

BASE_URL = "https://www.bailii.org"
CHUNK_CHAR_LIMIT = 4000  # Adjust for token limits (4k characters ~ 1000 tokens)

# Initialize Anthropic client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Comprehensive database paths for EW and UK
EW_DATABASES = [
    "/ew/cases/EWCA/Civ/", "/ew/cases/EWCA/Crim/", "/ew/cases/EWHC/Admin/",
    "/ew/cases/EWHC/Admlty/", "/ew/cases/EWHC/Ch/", "/ew/cases/EWHC/Comm/",
    "/ew/cases/EWHC/Costs/", "/ew/cases/EWHC/Exch/", "/ew/cases/EWHC/Fam/",
    "/ew/cases/EWHC/Mercantile/", "/ew/cases/EWHC/Patents/", "/ew/cases/EWHC/KB/",
    "/ew/cases/EWHC/QB/", "/ew/cases/EWHC/TCC/", "/ew/cases/EWPCC/",
    "/ew/cases/EWHC/IPEC/", "/ew/cases/EWCOP/", "/ew/cases/EWFC/",
    "/ew/cases/EWFC/HCJ/", "/ew/cases/EWFC/OJ/", "/ew/cases/EWMC/FPC/"
]

UK_DATABASES = [
    "/uk/cases/UKPC/", "/uk/cases/UKHL/", "/uk/cases/UKSC/",
    "/uk/cases/UKUT/AAC/", "/uk/cases/UKUT/TCC/", "/uk/cases/UKUT/IAC/",
    "/uk/cases/UKUT/LC/", "/uk/cases/UKFTT/GRC/", "/uk/cases/UKFTT/HESC/",
    "/uk/cases/UKFTT/PC/", "/uk/cases/UKFTT/TC/", "/uk/cases/CAT/",
    "/uk/cases/DRS/", "/uk/cases/SIAC/", "/uk/cases/UKEAT/",
    "/uk/cases/UKET/", "/uk/cases/UKFSM/", "/uk/cases/UKIAT/",
    "/uk/cases/UKAITUR/", "/uk/cases/UKIT/", "/uk/cases/UKSPC/",
    "/uk/cases/UKSSCSC/"
]

logger = logging.getLogger(__name__)

class BailiiScraper:
    def __init__(self, output_dir: str = "case_law", max_depth: int = 3, delay: float = 1.0, enable_pause_controls: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_depth = max_depth
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.case_urls: Set[str] = set()
        
        # Initialize pipeline controller
        self.controller = None
        if enable_pause_controls:
            try:
                self.controller = PipelineController()
                self.controller.register_callback('database_update', create_database_update_callback(self))
                self.controller.register_callback('dataset_creation', create_dataset_creation_callback(self))
                # Control message now handled by curses footer
            except Exception as e:
                logger.warning(f"Could not initialize pause controls: {e}")
                self.controller = None
        
        # Initialize rate limiter (10 requests per minute to be respectful)
        self.rate_limiter = RateLimiter(max_requests=10, time_window=60, delay_between_requests=1.0)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def is_case_url(self, url: str) -> bool:
        """Check if URL points to an actual case document"""
        return url.endswith('.html') and any(db_path in url for db_path in EW_DATABASES + UK_DATABASES)
    
    def is_valid_bailii_url(self, url: str) -> bool:
        """Check if URL is a valid bailii URL we want to crawl"""
        parsed = urlparse(url)
        return (parsed.netloc == 'www.bailii.org' and 
                (url.startswith(BASE_URL + '/ew/') or url.startswith(BASE_URL + '/uk/')))
    
    def get_all_links_from_page(self, url: str) -> List[str]:
        """Extract all relevant links from a page"""
        try:
            # Apply rate limiting before making the request
            self.rate_limiter.wait_if_needed()
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                full_url = urljoin(url, href)
                
                if self.is_valid_bailii_url(full_url):
                    links.append(full_url)
            
            return links
        except Exception as e:
            logger.error(f"Error getting links from {url}: {e}")
            return []
    
    def crawl_database_recursively(self, start_url: str) -> Set[str]:
        """Recursively crawl a database to find all case URLs"""
        queue = deque([(start_url, 0)])
        local_visited = set()
        case_urls = set()
        
        while queue:
            current_url, depth = queue.popleft()
            
            if (current_url in local_visited or 
                current_url in self.visited_urls or 
                depth > self.max_depth):
                continue
            
            local_visited.add(current_url)
            self.visited_urls.add(current_url)
            
            logger.info(f"Crawling depth {depth}: {current_url}")
            
            # Check for pause/quit commands
            if self.controller:
                command = self.controller.check_for_commands()
                if command == 'quit':
                    return case_urls
                self.controller.wait_while_paused()
                
                # Set current phase for pause state tracking
                self.controller.set_current_phase(f"Crawling database", {
                    'current_url': current_url,
                    'depth': depth,
                    'queue_size': len(queue),
                    'cases_found': len(case_urls)
                })
            
            if self.is_case_url(current_url):
                case_urls.add(current_url)
                logger.info(f"Found case: {current_url}")
                continue
            
            # Get all links from current page
            links = self.get_all_links_from_page(current_url)
            
            for link in links:
                if link not in local_visited and link not in self.visited_urls:
                    queue.append((link, depth + 1))
            
            time.sleep(self.delay)
        
        return case_urls
    
    def discover_all_case_urls(self) -> Set[str]:
        """Discover all case URLs from EW and UK databases"""
        all_case_urls = set()
        all_databases = EW_DATABASES + UK_DATABASES
        
        for i, db_path in enumerate(tqdm(all_databases, desc="Crawling databases")):
            db_url = BASE_URL + db_path
            logger.info(f"Starting crawl of database: {db_url}")
            
            # Check for pause/quit commands
            if self.controller:
                command = self.controller.check_for_commands()
                if command == 'quit':
                    break
                self.controller.wait_while_paused()
                
                # Set current phase for pause state tracking
                self.controller.set_current_phase(f"Discovering case URLs", {
                    'current_database': db_path,
                    'database_progress': f"{i+1}/{len(all_databases)}",
                    'total_cases_found': len(all_case_urls)
                })
            
            case_urls = self.crawl_database_recursively(db_url)
            all_case_urls.update(case_urls)
            
            logger.info(f"Found {len(case_urls)} cases in {db_path}")
        
        return all_case_urls
    
    def extract_case_content(self, html_content: str, url: str) -> Optional[Dict]:
        """Extract case content from HTML"""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text(separator="\n")
            lines = text.splitlines()
            
            # Basic case information
            case_data = {
                'url': url,
                'title': lines[0].strip() if lines else "Unknown Case",
                'content': text.strip(),
                'summary': ' '.join(lines[1:5]) if len(lines) > 1 else ""
            }
            
            return case_data
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None

    def process_case_urls(self, case_urls: Set[str], max_cases: Optional[int] = None) -> List[Dict]:
        """Process discovered case URLs and extract content"""
        all_cases = []
        case_list = list(case_urls)
        
        if max_cases:
            case_list = case_list[:max_cases]
        
        for url in tqdm(case_list, desc="Processing cases"):
            try:
                entries = self.extract_case_data_with_chunks(url)
                if entries:
                    all_cases.extend(entries)
                time.sleep(self.delay)
            except Exception as e:
                logger.error(f"Error processing case {url}: {e}")
                continue
        
        return all_cases
    
    def extract_case_data_with_chunks(self, url: str) -> List[Dict]:
        """Extract case data and create chunks for training"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n")
            lines = text.splitlines()
            case_name = lines[0].strip() if lines else "Unknown Case"
            
            # Clean and prepare text for Claude analysis
            clean_text = re.sub(r'\s+', ' ', text).strip()
            
            # Use Claude to analyze the case
            claude_analysis = self.analyze_case_with_claude(clean_text[:8000], case_name)
            
            # Create chunks for fine-tuning
            chunks = chunk_text(text)
            
            entries = []
            for i, chunk in enumerate(chunks):
                entries.append({
                    "instruction": "Analyze this UK legal case text and provide detailed legal reasoning, referenced legislation, and key principles.",
                    "input": chunk.strip(),
                    "output": f"Case: {case_name} (Part {i+1})\n\nClaude Analysis:\n{claude_analysis}"
                })
            return entries
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return []
    
    def analyze_case_with_claude(self, case_text: str, case_name: str) -> str:
        """Use Claude to analyze legal case text and extract key information."""
        try:
            prompt = f"""Analyze this UK legal case and provide a structured analysis:

Case Text:
{case_text}

Please provide:
1. Key legal issues and reasoning
2. Referenced legislation and legal precedents
3. Court's decision and rationale
4. Legal principles established or applied

Format your response as structured text suitable for fine-tuning data."""

            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text
        except Exception as e:
            logger.error(f"Error analyzing case with Claude: {e}")
            return f"Analysis unavailable for {case_name}"

def chunk_text(text, max_len=CHUNK_CHAR_LIMIT):
    return textwrap.wrap(text, max_len, break_long_words=False, replace_whitespace=False)

def main():
    """Main execution function for comprehensive BAILII scraping"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BAILII Case Law Scraper")
    parser.add_argument('--max-documents', type=int, help='Maximum number of cases to download (default: ALL)')
    parser.add_argument('--output-dir', default='bailii_cases', help='Output directory for scraped cases')
    
    args = parser.parse_args()
    
    scraper = BailiiScraper(max_depth=3, delay=1.0)
    
    # Control message now handled by curses footer
    
    try:
        logger.info("Starting comprehensive BAILII case discovery...")
        
        # Discover all case URLs
        case_urls = scraper.discover_all_case_urls()
        logger.info(f"Discovered {len(case_urls)} total case URLs")
        
        # Save discovered URLs for reference
        with open("discovered_case_urls.json", "w", encoding="utf-8") as f:
            json.dump(list(case_urls), f, indent=2)
        
        # Process cases
        max_cases = args.max_documents  # Use command line limit if provided, otherwise process ALL
        if max_cases:
            logger.info(f"Processing up to {max_cases} cases...")
        else:
            logger.info(f"Processing ALL {len(case_urls)} discovered cases...")
        
        all_cases = scraper.process_case_urls(case_urls, max_cases=max_cases)
        
        # Save the dataset
        output_file = "uk_legal_cases_comprehensive.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_cases, f, indent=2)
        
        logger.info(f"✅ Saved {len(all_cases)} fine-tuning entries to {output_file}")
        logger.info(f"✅ Total unique case URLs discovered: {len(case_urls)}")
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
    finally:
        # Cleanup controller
        if scraper.controller:
            scraper.controller.cleanup()

if __name__ == "__main__":
    main()
