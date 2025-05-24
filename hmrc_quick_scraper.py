#!/usr/bin/env python3
"""
Quick HMRC Scraper - Limited discovery for faster results
"""

import sys
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.hmrc_scraper import HMRCScraper
from utils.hmrc_curses_wrapper import run_hmrc_scraper_with_curses
from utils.rate_limiter import RateLimiter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_quick_hmrc_scraper(output_dir="hmrc_documentation_quick", max_batches=20, max_documents=100):
    """Run HMRC scraper with limited discovery for quick results"""
    
    # Progress tracking
    progress_data = {
        'discovered': 0,
        'downloaded': 0,
        'status': 'Initializing...'
    }
    
    def progress_callback(data):
        """Update progress data"""
        progress_data.update(data)
        logger.info(f"Progress: {data.get('status', 'Working...')} - Discovered: {data.get('discovered', 0)}, Downloaded: {data.get('downloaded', 0)}")
    
    try:
        # Create scraper with progress callback
        scraper = HMRCScraper(output_dir, progress_callback=progress_callback)
        
        logger.info(f"Starting quick HMRC scrape with max_batches={max_batches}, max_documents={max_documents}")
        
        # Load existing progress
        scraper.load_progress()
        
        # Run limited discovery
        logger.info("Running limited discovery...")
        all_urls = scraper.discover_via_search_api(max_batches=max_batches)
        scraper.discovered_urls = all_urls
        
        logger.info(f"Discovered {len(all_urls)} documents in {max_batches} batches")
        
        # Save discovery results
        scraper.save_progress()
        
        # Download documents
        if len(all_urls) > 0:
            logger.info(f"Starting download of up to {max_documents} documents...")
            scraper.download_all_documents(max_documents=max_documents)
            
            # Generate summary
            summary = scraper.generate_summary()
            
            logger.info("=== QUICK HMRC SCRAPE COMPLETE ===")
            logger.info(f"Discovered: {summary['total_discovered']}")
            logger.info(f"Downloaded: {summary['total_downloaded']}")
            logger.info(f"Failed: {summary['total_failed']}")
            
            return {
                'status': 'success',
                'discovered': summary['total_discovered'],
                'downloaded': summary['total_downloaded'],
                'failed': summary['total_failed']
            }
        else:
            logger.warning("No documents discovered!")
            return {
                'status': 'error',
                'error': 'No documents discovered',
                'discovered': 0,
                'downloaded': 0
            }
            
    except Exception as e:
        logger.error(f"Error during quick scrape: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'discovered': progress_data['discovered'],
            'downloaded': progress_data['downloaded']
        }

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick HMRC Scraper - Limited discovery for faster results")
    parser.add_argument('--output-dir', default='hmrc_documentation_quick',
                       help='Directory to store HMRC documentation')
    parser.add_argument('--max-batches', type=int, default=20,
                       help='Maximum number of discovery batches (default: 20)')
    parser.add_argument('--max-documents', type=int, default=100,
                       help='Maximum number of documents to download (default: 100)')
    parser.add_argument('--no-curses', action='store_true',
                       help='Run without curses interface')
    
    args = parser.parse_args()
    
    if args.no_curses:
        # Run directly without curses
        result = run_quick_hmrc_scraper(
            output_dir=args.output_dir,
            max_batches=args.max_batches,
            max_documents=args.max_documents
        )
        
        if result['status'] == 'success':
            print(f"\n✓ Quick HMRC scrape completed successfully!")
            print(f"  Discovered: {result['discovered']} documents")
            print(f"  Downloaded: {result['downloaded']} documents")
            print(f"  Failed: {result.get('failed', 0)} documents")
        else:
            print(f"\n✗ Quick HMRC scrape failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    else:
        # Run with curses interface
        run_hmrc_scraper_with_curses(
            lambda: run_quick_hmrc_scraper(
                output_dir=args.output_dir,
                max_batches=args.max_batches,
                max_documents=args.max_documents
            )
        )

if __name__ == "__main__":
    main()