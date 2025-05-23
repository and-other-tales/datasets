#!/usr/bin/env python3
"""
Housing-Specific Bailli Case Law Scraper
Scrapes housing, tenancy, landlord-tenant, and eviction cases from Bailli
"""

import re
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.pipeline_controller import PipelineController, create_database_update_callback, create_dataset_creation_callback
from pipelines.bailii_scraper import BailiiScraper

logger = logging.getLogger(__name__)

class HousingBailiiScraper(BailiiScraper):
    def __init__(self, output_dir: str = "housing_case_law"):
        super().__init__(output_dir, enable_pause_controls=True)
        
        # Initialize additional pipeline controller for housing-specific features
        self.housing_controller = PipelineController()
        
        # Register pause functionality callbacks
        self.housing_controller.register_callback('database_update', create_database_update_callback(self))
        self.housing_controller.register_callback('dataset_creation', create_dataset_creation_callback(self))
        
        logger.info("🔶 Pipeline Control: Press P to pause/resume, A to update databases (when paused), D to create dataset (when paused), Q to quit")
        
        # Housing-specific case law keywords
        self.housing_keywords = {
            'tenancy_types': [
                'assured shorthold tenancy', 'assured tenancy', 'secure tenancy',
                'protected tenancy', 'statutory tenancy', 'periodic tenancy',
                'fixed term tenancy', 'joint tenancy', 'sole tenancy',
                'residential tenancy', 'commercial tenancy', 'business tenancy'
            ],
            'eviction_possession': [
                'possession', 'eviction', 'notice to quit', 'section 8',
                'section 21', 'mandatory ground', 'discretionary ground',
                'possession order', 'possession proceedings', 'accelerated possession',
                'warrant of possession', 'bailiff', 'enforcement',
                'notice seeking possession', 'ground for possession',
                'suitable alternative accommodation', 'rent arrears',
                'breach of tenancy', 'anti-social behaviour'
            ],
            'housing_rights': [
                'quiet enjoyment', 'repair obligations', 'housing standards',
                'fitness for habitation', 'overcrowding', 'harassment',
                'illegal eviction', 'retaliatory eviction', 'discriminatory eviction',
                'right to buy', 'right to acquire', 'preserved right to buy',
                'tenant rights', 'landlord obligations', 'housing benefit',
                'universal credit', 'deposit protection'
            ],
            'property_issues': [
                'disrepair', 'condensation', 'damp', 'mould', 'heating',
                'hot water', 'structural defects', 'electrical safety',
                'gas safety', 'fire safety', 'energy efficiency',
                'houses in multiple occupation', 'hmo', 'licensing'
            ],
            'leasehold_issues': [
                'service charge', 'ground rent', 'major works',
                'section 20', 'consultation', 'reserve fund',
                'administration charge', 'lease extension',
                'enfranchisement', 'right to manage', 'forfeiture'
            ],
            'primary_terms': [
                'housing', 'tenant', 'tenancy', 'landlord', 'rent', 'rental',
                'leasehold', 'freehold', 'eviction', 'deposit', 'letting',
                'residential', 'dwelling', 'premises', 'property'
            ]
        }
        
        # Specific housing courts and tribunals
        self.housing_courts = [
            'First-tier Tribunal (Property Chamber)',
            'Upper Tribunal (Lands Chamber)',
            'County Court',
            'High Court',
            'Court of Appeal',
            'Supreme Court'
        ]
        
        # Housing-specific search terms for Bailli
        self.housing_search_terms = [
            'housing', 'tenancy', 'landlord', 'tenant', 'rent', 'eviction',
            'possession', 'assured shorthold', 'section 21', 'section 8',
            'deposit', 'leasehold', 'service charge', 'disrepair',
            'harassment', 'illegal eviction', 'notice to quit'
        ]
    
    def is_housing_case(self, case_title: str, case_summary: str = "", case_text: str = "") -> bool:
        """Check if a case is housing-related"""
        combined_text = (case_title + " " + case_summary + " " + case_text[:1000]).lower()
        
        # Check for housing-specific terms
        all_housing_terms = []
        for category in self.housing_keywords.values():
            all_housing_terms.extend(category)
        
        housing_matches = sum(1 for term in all_housing_terms if term in combined_text)
        
        # More specific criteria for housing cases
        return (
            housing_matches >= 3 or
            any(term in combined_text for term in [
                'housing', 'tenancy', 'landlord', 'eviction', 'possession',
                'assured shorthold', 'section 21', 'section 8'
            ]) or
            'first-tier tribunal (property chamber)' in combined_text or
            'upper tribunal (lands chamber)' in combined_text
        )
    
    def search_housing_cases_by_court(self, court: str, max_pages: int = 50) -> List[Dict]:
        """Search for housing cases in a specific court"""
        housing_cases = []
        
        logger.info(f"Searching {court} for housing cases...")
        
        # Search each housing term in the court
        for search_term in self.housing_search_terms:
            try:
                # Construct search URL for the court and term
                search_url = f"https://www.bailii.org/cgi-bin/markup.cgi?doc=/databases/{court.lower().replace(' ', '_')}/&query={search_term}"
                
                # Get search results
                response = self.session.get(search_url, timeout=30)
                if response.status_code == 200:
                    # Parse case links from results
                    case_links = re.findall(
                        r'<a href="([^"]+)"[^>]*>([^<]+)</a>',
                        response.text
                    )
                    
                    for link, title in case_links:
                        if self.is_housing_case(title):
                            full_url = f"https://www.bailii.org{link}" if link.startswith('/') else link
                            
                            case_info = {
                                'url': full_url,
                                'title': title.strip(),
                                'court': court,
                                'search_term': search_term
                            }
                            
                            housing_cases.append(case_info)
                            logger.info(f"Found housing case: {title}")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error searching {court} for {search_term}: {e}")
                continue
        
        return housing_cases
    
    def search_property_tribunal_cases(self) -> List[Dict]:
        """Search First-tier Tribunal (Property Chamber) cases"""
        logger.info("Searching First-tier Tribunal (Property Chamber) for housing cases...")
        
        property_cases = []
        
        # Property Chamber has specific categories
        property_categories = [
            'residential_property',
            'leasehold',
            'park_homes',
            'land_registration',
            'housing_act'
        ]
        
        for category in property_categories:
            try:
                # Search in property chamber database
                base_url = "https://www.bailii.org/uk/cases/UKFTT/"
                
                # Get recent cases from this category
                response = self.session.get(f"{base_url}{category}/", timeout=30)
                if response.status_code == 200:
                    case_links = re.findall(
                        r'<a href="([^"]+\.html)"[^>]*>([^<]+)</a>',
                        response.text
                    )
                    
                    for link, title in case_links:
                        if self.is_housing_case(title):
                            full_url = f"{base_url}{category}/{link}"
                            
                            case_info = {
                                'url': full_url,
                                'title': title.strip(),
                                'court': 'First-tier Tribunal (Property Chamber)',
                                'category': category
                            }
                            
                            property_cases.append(case_info)
                            logger.info(f"Found property tribunal case: {title}")
                
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Error searching property chamber {category}: {e}")
                continue
        
        return property_cases
    
    def discover_housing_cases(self, max_cases_per_court: int = 100) -> List[Dict]:
        """Discover housing cases across all relevant courts"""
        logger.info("=== DISCOVERING HOUSING CASES ===")
        
        all_housing_cases = []
        
        # Search property tribunal first (most relevant)
        property_cases = self.search_property_tribunal_cases()
        all_housing_cases.extend(property_cases[:max_cases_per_court])
        
        # Search other courts
        other_courts = [
            'EWCA',  # Court of Appeal
            'EWHC',  # High Court
            'UKUT',  # Upper Tribunal
            'UKSC'   # Supreme Court
        ]
        
        for court in other_courts:
            court_cases = self.search_housing_cases_by_court(court, max_pages=10)
            all_housing_cases.extend(court_cases[:max_cases_per_court//4])
        
        # Remove duplicates
        seen_urls = set()
        unique_cases = []
        for case in all_housing_cases:
            if case['url'] not in seen_urls:
                seen_urls.add(case['url'])
                unique_cases.append(case)
        
        logger.info(f"Discovered {len(unique_cases)} unique housing cases")
        return unique_cases
    
    def scrape_housing_case(self, case_info: Dict) -> Optional[Dict]:
        """Scrape a single housing case"""
        try:
            response = self.session.get(case_info['url'], timeout=30)
            if response.status_code == 200:
                # Extract case content using parent class method
                case_data = self.extract_case_content(response.text, case_info['url'])
                
                if case_data and self.is_housing_case(
                    case_data.get('title', ''), 
                    case_data.get('summary', ''), 
                    case_data.get('content', '')
                ):
                    # Add housing-specific metadata
                    case_data.update({
                        'court': case_info.get('court', ''),
                        'category': case_info.get('category', ''),
                        'search_term': case_info.get('search_term', ''),
                        'case_type': 'housing'
                    })
                    
                    return case_data
            
        except Exception as e:
            logger.error(f"Error scraping case {case_info['url']}: {e}")
        
        return None
    
    def scrape_all_housing_cases(self, max_cases: int = 500) -> List[Dict]:
        """Scrape all discovered housing cases"""
        # Discover cases first
        housing_cases = self.discover_housing_cases()
        
        if not housing_cases:
            logger.warning("No housing cases discovered")
            return []
        
        # Limit number of cases
        housing_cases = housing_cases[:max_cases]
        
        logger.info(f"Scraping {len(housing_cases)} housing cases...")
        
        scraped_cases = []
        
        for i, case_info in enumerate(housing_cases):
            logger.info(f"Scraping case {i+1}/{len(housing_cases)}: {case_info['title']}")
            
            case_data = self.scrape_housing_case(case_info)
            if case_data:
                scraped_cases.append(case_data)
            
            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(f"Scraped {len(scraped_cases)}/{i+1} successful cases")
            
            time.sleep(2)  # Rate limiting
        
        logger.info(f"Successfully scraped {len(scraped_cases)} housing cases")
        return scraped_cases
    
    def save_housing_cases(self, cases: List[Dict], filename: str = "housing_cases.json"):
        """Save housing cases to file"""
        output_path = Path(self.output_dir) / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(cases)} housing cases to {output_path}")
        
        # Generate summary
        self.generate_housing_case_summary(cases)
    
    def generate_housing_case_summary(self, cases: List[Dict]):
        """Generate summary of housing cases"""
        summary = {
            'total_cases': len(cases),
            'courts': {},
            'categories': {},
            'housing_topics': {},
            'date_range': {}
        }
        
        # Analyze cases
        for case in cases:
            # Court distribution
            court = case.get('court', 'Unknown')
            summary['courts'][court] = summary['courts'].get(court, 0) + 1
            
            # Category distribution
            category = case.get('category', 'Unknown')
            summary['categories'][category] = summary['categories'].get(category, 0) + 1
            
            # Housing topic analysis
            content = (case.get('title', '') + ' ' + case.get('content', '')).lower()
            for topic_category, terms in self.housing_keywords.items():
                for term in terms:
                    if term in content:
                        summary['housing_topics'][term] = summary['housing_topics'].get(term, 0) + 1
        
        # Save summary
        summary_path = Path(self.output_dir) / "housing_case_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Housing case summary saved to {summary_path}")

def main():
    """Main function for housing case law scraping"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape housing-related case law from Bailli")
    parser.add_argument('--output-dir', default='housing_case_law',
                       help='Directory to store housing case law')
    parser.add_argument('--max-cases', type=int, default=500,
                       help='Maximum number of cases to scrape')
    parser.add_argument('--discover-only', action='store_true',
                       help='Only discover cases, do not scrape content')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    scraper = HousingBailiiScraper(args.output_dir)
    
    try:
        if args.discover_only:
            cases = scraper.discover_housing_cases()
            print(f"Discovered {len(cases)} housing cases")
        else:
            cases = scraper.scrape_all_housing_cases(args.max_cases)
            scraper.save_housing_cases(cases)
            
            print(f"\n=== HOUSING CASE LAW SCRAPING COMPLETE ===")
            print(f"Total cases scraped: {len(cases)}")
            print(f"Output directory: {args.output_dir}")
            
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")

if __name__ == "__main__":
    main()