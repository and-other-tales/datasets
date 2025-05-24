#!/usr/bin/env python3
"""
HMRC Scraper Runner - Optimized for menu system
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.hmrc_scraper import HMRCScraper
from utils.rate_limiter import RateLimiter

def run_hmrc_scraper():
    """Run HMRC scraper with sensible defaults"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('hmrc_scraper.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Configuration
    output_dir = "hmrc_documentation"
    max_batches = 10  # Limit discovery to 10 batches (1000 documents) for reasonable speed
    max_documents = 50  # Download 50 documents as a good starting point
    
    try:
        logger.info("=== HMRC Documentation Scraper ===")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Discovery limit: {max_batches} batches")
        logger.info(f"Download limit: {max_documents} documents")
        
        # Create scraper
        scraper = HMRCScraper(output_dir)
        
        # Load existing progress
        logger.info("Loading existing progress...")
        scraper.load_progress()
        
        if len(scraper.downloaded_urls) > 0:
            logger.info(f"Found {len(scraper.downloaded_urls)} previously downloaded documents")
        
        # Run discovery with limit
        logger.info("Starting discovery phase...")
        all_urls = scraper.discover_via_search_api(max_batches=max_batches)
        scraper.discovered_urls = all_urls
        
        logger.info(f"Discovery complete: Found {len(all_urls)} relevant documents")
        
        # Save discovery results
        scraper.save_progress()
        
        # Download documents
        if len(all_urls) > 0:
            logger.info("Starting download phase...")
            scraper.download_all_documents(max_documents=max_documents)
            
            # Generate summary
            summary = scraper.generate_summary()
            
            logger.info("\n=== SUMMARY ===")
            logger.info(f"Total discovered: {summary['total_discovered']}")
            logger.info(f"Total downloaded: {summary['total_downloaded']}")
            logger.info(f"Total failed: {summary['total_failed']}")
            
            # Show category breakdown
            if summary.get('tax_categories'):
                logger.info("\nDocuments by tax category:")
                for category, count in summary['tax_categories'].items():
                    logger.info(f"  {category}: {count}")
            
            logger.info(f"\nOutput saved to: {output_dir}/")
            
            return True
        else:
            logger.error("No documents discovered!")
            return False
            
    except KeyboardInterrupt:
        logger.info("\nScraping interrupted by user")
        if 'scraper' in locals():
            scraper.save_progress()
        return False
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        if 'scraper' in locals():
            scraper.save_progress()
        return False

if __name__ == "__main__":
    success = run_hmrc_scraper()
    if success:
        print("\n✓ HMRC scraping completed successfully!")
    else:
        print("\n✗ HMRC scraping failed or was interrupted")
        sys.exit(1)