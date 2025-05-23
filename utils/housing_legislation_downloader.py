#!/usr/bin/env python3
"""
Housing-Specific UK Legislation Downloader
Downloads only housing, tenancy, and landlord-tenant related legislation
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from improved_downloader import ImprovedUKLegislationDownloader

logger = logging.getLogger(__name__)

class HousingLegislationDownloader(ImprovedUKLegislationDownloader):
    def __init__(self, output_dir: str = "housing_legislation"):
        super().__init__(output_dir)
        
        # Initialize URL tracking sets
        self.discovered_urls = set()
        self.downloaded_urls = set()
        
        # Housing-specific keywords for filtering
        self.housing_keywords = {
            'primary_terms': [
                'housing', 'tenant', 'tenancy', 'landlord', 'rent', 'rental',
                'leasehold', 'freehold', 'eviction', 'deposit', 'tenants',
                'landlords', 'renting', 'lettings', 'dwellings', 'residential'
            ],
            'tenancy_terms': [
                'assured shorthold tenancy', 'assured tenancy', 'secure tenancy',
                'protected tenancy', 'statutory tenancy', 'periodic tenancy',
                'fixed term tenancy', 'joint tenancy', 'sole tenancy',
                'subletting', 'assignment', 'succession', 'tenancy agreement',
                'lease agreement', 'rental agreement', 'letting agreement'
            ],
            'eviction_terms': [
                'eviction', 'possession', 'notice to quit', 'section 8',
                'section 21', 'mandatory ground', 'discretionary ground',
                'possession order', 'possession proceedings', 'accelerated possession',
                'warrant of possession', 'bailiff', 'enforcement',
                'notice seeking possession', 'ground for possession',
                'suitable alternative accommodation', 'rent arrears'
            ],
            'housing_rights': [
                'right to buy', 'right to acquire', 'preserved right to buy',
                'tenant rights', 'landlord obligations', 'quiet enjoyment',
                'repair obligations', 'housing standards', 'fitness for habitation',
                'overcrowding', 'harassment', 'illegal eviction',
                'retaliatory eviction', 'discriminatory eviction'
            ],
            'housing_acts': [
                'housing act', 'rent act', 'landlord and tenant act', 
                'tenancy act', 'leasehold reform act', 'homelessness act',
                'right to buy', 'assured tenancy', 'secure tenancy',
                'housing benefit', 'universal credit', 'tenant fees act',
                'protection from eviction act', 'rent assessment committee'
            ],
            'specific_legislation': [
                'housing act 1985', 'housing act 1988', 'housing act 1996',
                'housing act 2004', 'rent act 1977', 'landlord and tenant act 1985',
                'landlord and tenant act 1987', 'leasehold reform act 1967',
                'leasehold reform housing and urban development act 1993',
                'commonhold and leasehold reform act 2002',
                'housing and planning act 2016', 'tenant fees act 2019',
                'renters rights bill', 'homelessness reduction act 2017',
                'protection from eviction act 1977', 'defective premises act 1972',
                'mobile homes act', 'caravan sites act', 'houses in multiple occupation'
            ]
        }
        
        # Exclude non-housing related terms
        self.exclude_terms = [
            'criminal', 'police', 'immigration', 'tax', 'company',
            'employment', 'education', 'health', 'transport', 'planning',
            'environmental', 'planning permission', 'building regulations'
        ]
        
    def is_housing_related(self, title: str, summary: str = "") -> bool:
        """Check if legislation is housing-related"""
        text = (title + " " + summary).lower()
        
        # Exclude clearly non-housing legislation
        for exclude_term in self.exclude_terms:
            if exclude_term in text and not any(housing_term in text for housing_term in self.housing_keywords['primary_terms']):
                return False
        
        # Check for specific housing legislation
        for specific_act in self.housing_keywords['specific_legislation']:
            if specific_act in text:
                return True
        
        # Check for housing acts
        for housing_act in self.housing_keywords['housing_acts']:
            if housing_act in text:
                return True
        
        # Check for any housing-related terms
        all_housing_terms = (
            self.housing_keywords['primary_terms'] + 
            self.housing_keywords['tenancy_terms'] + 
            self.housing_keywords['eviction_terms'] + 
            self.housing_keywords['housing_rights']
        )
        
        housing_term_count = sum(1 for term in all_housing_terms if term in text)
        
        # More lenient matching for housing-related content
        return (
            housing_term_count >= 2 or 
            any(term in text for term in ['housing', 'tenancy', 'landlord', 'eviction', 'possession']) or
            any(term in text for term in self.housing_keywords['eviction_terms'])
        )
    
    def discover_housing_legislation(self):
        """Discover housing-specific legislation"""
        logger.info("Discovering housing-related legislation...")
        
        # Base URLs for different types of housing legislation
        housing_search_terms = [
            "housing", "tenancy", "landlord", "rent", "leasehold", "tenant",
            "residential", "dwelling", "eviction", "possession", "letting",
            "assured", "shorthold", "deposit", "bailiff", "harassment"
        ]
        
        discovered_count = 0
        
        for search_term in housing_search_terms:
            logger.info(f"Searching for legislation containing '{search_term}'...")
            
            # Search through different categories
            for category in ['ukpga', 'uksi', 'asp', 'asc']:
                try:
                    # Use the existing discovery mechanism but filter results
                    category_url = f"https://www.legislation.gov.uk/{category}"
                    
                    # Get list of years for this category
                    years_response = self.session.get(category_url, timeout=30)
                    if years_response.status_code == 200:
                        # Extract years and search through them
                        years = re.findall(r'/(\d{4})/', years_response.text)
                        years = sorted(set(years), reverse=True)[:20]  # Last 20 years
                        
                        for year in years:
                            year_url = f"{category_url}/{year}"
                            year_response = self.session.get(year_url, timeout=30)
                            
                            if year_response.status_code == 200:
                                # Find housing-related acts
                                act_links = re.findall(
                                    r'<a[^>]+href="(/[^"]+/\d+/[^"]+)"[^>]*>([^<]+)</a>',
                                    year_response.text
                                )
                                
                                for link, title in act_links:
                                    if self.is_housing_related(title):
                                        full_url = f"https://www.legislation.gov.uk{link}"
                                        if full_url not in self.discovered_urls:
                                            self.discovered_urls.add(full_url)
                                            discovered_count += 1
                                            logger.info(f"Found housing legislation: {title}")
                            
                            time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error searching {category} for {search_term}: {e}")
                    continue
        
        logger.info(f"Discovered {discovered_count} housing-related legislation items")
        self.save_progress()
        
    def download_specific_housing_acts(self):
        """Download well-known housing acts"""
        specific_housing_urls = [
            # Core Housing Acts
            "https://www.legislation.gov.uk/ukpga/1985/68",  # Housing Act 1985
            "https://www.legislation.gov.uk/ukpga/1988/50",  # Housing Act 1988
            "https://www.legislation.gov.uk/ukpga/1996/52",  # Housing Act 1996
            "https://www.legislation.gov.uk/ukpga/2004/34",  # Housing Act 2004
            
            # Rent Acts
            "https://www.legislation.gov.uk/ukpga/1977/42",  # Rent Act 1977
            "https://www.legislation.gov.uk/ukpga/1965/75",  # Rent Act 1965
            
            # Landlord and Tenant Acts
            "https://www.legislation.gov.uk/ukpga/1985/70",  # Landlord and Tenant Act 1985
            "https://www.legislation.gov.uk/ukpga/1987/31",  # Landlord and Tenant Act 1987
            "https://www.legislation.gov.uk/ukpga/1954/56",  # Landlord and Tenant Act 1954
            
            # Leasehold Acts
            "https://www.legislation.gov.uk/ukpga/1967/88",  # Leasehold Reform Act 1967
            "https://www.legislation.gov.uk/ukpga/1993/28",  # Leasehold Reform, Housing and Urban Development Act 1993
            "https://www.legislation.gov.uk/ukpga/2002/15",  # Commonhold and Leasehold Reform Act 2002
            
            # Recent Housing Legislation
            "https://www.legislation.gov.uk/ukpga/2016/22",  # Housing and Planning Act 2016
            "https://www.legislation.gov.uk/ukpga/2019/4",   # Tenant Fees Act 2019
            "https://www.legislation.gov.uk/ukpga/2017/13",  # Homelessness Reduction Act 2017
            
            # Protection and Rights
            "https://www.legislation.gov.uk/ukpga/1977/43",  # Protection from Eviction Act 1977
            "https://www.legislation.gov.uk/ukpga/1972/35",  # Defective Premises Act 1972
            
            # Mobile Homes and Caravans
            "https://www.legislation.gov.uk/ukpga/2013/5",   # Mobile Homes Act 2013
            "https://www.legislation.gov.uk/ukpga/1983/34",  # Mobile Homes Act 1983
            "https://www.legislation.gov.uk/ukpga/1968/52",  # Caravan Sites Act 1968
            
            # Additional Housing-Related Acts
            "https://www.legislation.gov.uk/ukpga/2020/17",  # Coronavirus Act 2020 (includes eviction provisions)
            "https://www.legislation.gov.uk/ukpga/1988/9",   # Local Government Finance Act 1988 (Council Tax)
            "https://www.legislation.gov.uk/ukpga/2014/6",   # Deregulation Act 2014 (includes housing provisions)
        ]
        
        # Also include amendments and related SIs
        housing_amendments = [
            # Housing Act 1988 amendments
            "https://www.legislation.gov.uk/uksi/1988/2203",  # Assured Tenancies and Agricultural Occupancies Order 1988
            "https://www.legislation.gov.uk/uksi/1997/194",   # Assured Tenancies Order 1997
            
            # Housing Act 2004 amendments
            "https://www.legislation.gov.uk/uksi/2006/2825",  # Housing Health and Safety Rating System Regulations 2005
            "https://www.legislation.gov.uk/uksi/2007/1903",  # Houses in Multiple Occupation Regulations 2006
            
            # Tenant Fees Act amendments
            "https://www.legislation.gov.uk/uksi/2019/947",   # Tenant Fees Act 2019 (Relevant Persons) Regulations 2019
            
            # Recent eviction and possession amendments
            "https://www.legislation.gov.uk/uksi/2020/914",   # Coronavirus Act 2020 (Residential Tenancies) Regulations
            "https://www.legislation.gov.uk/uksi/2021/15",    # Assured Shorthold Tenancies (Amendment) Regulations 2021
        ]
        
        all_housing_urls = specific_housing_urls + housing_amendments
        
        logger.info("Adding specific housing acts and amendments to download queue...")
        
        for url in all_housing_urls:
            if url not in self.discovered_urls:
                self.discovered_urls.add(url)
                logger.info(f"Added housing legislation: {url}")
        
        logger.info(f"Added {len(all_housing_urls)} specific housing acts and amendments")
        self.save_progress()
    
    def run_housing_discovery(self):
        """Run complete housing legislation discovery"""
        logger.info("=== STARTING HOUSING LEGISLATION DISCOVERY ===")
        
        # Load any existing progress
        self.load_progress()
        
        # Add specific housing acts first
        self.download_specific_housing_acts()
        
        # Discover additional housing legislation
        self.discover_housing_legislation()
        
        logger.info(f"Total housing legislation discovered: {len(self.discovered_urls)}")
        
        # Save final progress
        self.save_progress()
        
    def download_all_housing_legislation(self):
        """Download all discovered housing legislation"""
        if not self.discovered_urls:
            logger.warning("No housing legislation discovered. Running discovery first...")
            self.run_housing_discovery()
        
        logger.info(f"Starting download of {len(self.discovered_urls)} housing legislation items...")
        
        # Use the parent class download method
        self.download_all_legislation()
        
    def generate_housing_summary(self):
        """Generate summary of downloaded housing legislation"""
        summary = {
            'total_discovered': len(self.discovered_urls),
            'total_downloaded': len(self.downloaded_urls),
            'housing_categories': {},
            'key_acts': []
        }
        
        # Categorize by type
        for url in self.downloaded_urls:
            if 'housing-act' in url.lower():
                category = 'Housing Acts'
            elif 'landlord-and-tenant' in url.lower():
                category = 'Landlord & Tenant Acts'
            elif 'rent-act' in url.lower():
                category = 'Rent Acts'
            elif 'leasehold' in url.lower():
                category = 'Leasehold Acts'
            elif 'tenancy' in url.lower():
                category = 'Tenancy Acts'
            else:
                category = 'Other Housing Legislation'
            
            summary['housing_categories'][category] = summary['housing_categories'].get(category, 0) + 1
        
        # Save summary
        summary_path = Path(self.output_dir) / "housing_legislation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Housing legislation summary saved to {summary_path}")
        return summary

def main():
    """Main function for housing legislation download"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download housing-related UK legislation")
    parser.add_argument('--output-dir', default='housing_legislation',
                       help='Directory to store housing legislation')
    parser.add_argument('--discover-only', action='store_true',
                       help='Only discover URLs, do not download')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    downloader = HousingLegislationDownloader(args.output_dir)
    
    try:
        if args.discover_only:
            downloader.run_housing_discovery()
        else:
            downloader.run_housing_discovery()
            downloader.download_all_housing_legislation()
            summary = downloader.generate_housing_summary()
            
            print(f"\n=== HOUSING LEGISLATION DOWNLOAD COMPLETE ===")
            print(f"Total items discovered: {summary['total_discovered']}")
            print(f"Total items downloaded: {summary['total_downloaded']}")
            print(f"Categories found: {list(summary['housing_categories'].keys())}")
            
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Download failed: {e}")

if __name__ == "__main__":
    main()