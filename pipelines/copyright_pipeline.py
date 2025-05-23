#!/usr/bin/env python3
"""
Complete Copyright Law Pipeline for LLM Training

This script runs the entire process from downloading copyright-specific UK legislation 
and case law to creating specialised datasets for copyright law LLM fine-tuning.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.pipeline_controller import PipelineController, create_database_update_callback, create_dataset_creation_callback

# Import our copyright-specific modules
from utils.copyright_legislation_downloader import CopyrightLegislationDownloader

# Conditional imports for modules that might require additional dependencies
try:
    from pipelines.copyright_bailii_scraper import CopyrightBailiiScraper
except ImportError:
    CopyrightBailiiScraper = None

try:
    from pipelines.copyright_QA_generator import CopyrightQAGenerator
except ImportError:
    CopyrightQAGenerator = None

try:
    from utils.dataset_creator import UKLegislationDatasetCreator
except ImportError:
    UKLegislationDatasetCreator = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/copyright_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CopyrightLawPipeline:
    def __init__(
        self,
        legislation_dir: str = "generated/copyright_legislation",
        case_law_dir: str = "generated/copyright_case_law",
        datasets_dir: str = "generated/copyright_datasets",
        max_legislation_items: Optional[int] = None,
        max_cases: Optional[int] = None
    ):
        """
        Initialise the Copyright Law Pipeline
        
        Args:
            legislation_dir: Directory for copyright legislation
            case_law_dir: Directory for copyright case law
            datasets_dir: Directory for final datasets
            max_legislation_items: Maximum legislation items to download
            max_cases: Maximum cases to download from BAILII
        """
        self.legislation_dir = Path(legislation_dir)
        self.case_law_dir = Path(case_law_dir)
        self.datasets_dir = Path(datasets_dir)
        self.max_legislation_items = max_legislation_items
        self.max_cases = max_cases
        
        # Initialize pipeline controller for pause functionality
        self.controller = PipelineController()
        
        # Register pause functionality callbacks
        self.controller.register_callback('database_update', create_database_update_callback(self))
        self.controller.register_callback('dataset_creation', create_dataset_creation_callback(self))
        
        # Create directories
        self.legislation_dir.mkdir(parents=True, exist_ok=True)
        self.case_law_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Legal Llama Copyright Pipeline initialised")
        logger.info(f"Legislation directory: {self.legislation_dir}")
        logger.info(f"Case law directory: {self.case_law_dir}")
        logger.info(f"Datasets directory: {self.datasets_dir}")
        logger.info("🔶 Pipeline Control: Press P to pause/resume, A to update databases (when paused), D to create dataset (when paused), Q to quit")
    
    def download_copyright_legislation(self) -> bool:
        """Download copyright-specific UK legislation using ParaLlama utilities"""
        logger.info("=== PHASE 1: ParaLlama Copyright Legislation Download ===")
        
        try:
            downloader = CopyrightLegislationDownloader(
                output_dir=str(self.legislation_dir),
                max_items=self.max_legislation_items
            )
            
            # Run the copyright legislation download
            success = downloader.run_copyright_download()
            
            if success:
                logger.info("ParaLlama copyright legislation download completed successfully")
                return True
            else:
                logger.error("ParaLlama copyright legislation download failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in ParaLlama copyright legislation download: {e}")
            return False
    
    def scrape_copyright_case_law(self) -> bool:
        """Scrape copyright-specific case law from BAILII using ParaLlama scraper"""
        logger.info("=== PHASE 2: ParaLlama Copyright Case Law Scraping ===")
        
        if CopyrightBailiiScraper is None:
            logger.warning("ParaLlama CopyrightBailiiScraper not available, skipping case law scraping")
            return True
        
        try:
            scraper = CopyrightBailiiScraper(
                output_dir=str(self.case_law_dir),
                max_cases=self.max_cases
            )
            
            # Run the copyright case law scraping
            success = scraper.scrape_copyright_cases()
            
            if success:
                logger.info("ParaLlama copyright case law scraping completed successfully")
                return True
            else:
                logger.error("ParaLlama copyright case law scraping failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in ParaLlama copyright case law scraping: {e}")
            return False
    
    def generate_copyright_qa_pairs(self) -> bool:
        """Generate copyright-specific Q&A pairs using ParaLlama QA generator"""
        logger.info("=== PHASE 3: ParaLlama Copyright Q&A Generation ===")
        
        if CopyrightQAGenerator is None:
            logger.warning("ParaLlama CopyrightQAGenerator not available, skipping Q&A generation")
            return True
        
        try:
            qa_generator = CopyrightQAGenerator(
                legislation_dir=str(self.legislation_dir),
                case_law_dir=str(self.case_law_dir),
                output_dir=str(self.datasets_dir / "qa_pairs")
            )
            
            # Generate copyright-specific Q&A pairs
            success = qa_generator.generate_copyright_qa()
            
            if success:
                logger.info("ParaLlama copyright Q&A generation completed successfully")
                return True
            else:
                logger.error("ParaLlama copyright Q&A generation failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in ParaLlama copyright Q&A generation: {e}")
            return False
    
    def create_copyright_datasets(self) -> bool:
        """Create final datasets for copyright law LLM training using ParaLlama tools"""
        logger.info("=== PHASE 4: ParaLlama Copyright Dataset Creation ===")
        
        if UKLegislationDatasetCreator is None:
            logger.warning("ParaLlama UKLegislationDatasetCreator not available, skipping dataset creation")
            return True
        
        try:
            dataset_creator = UKLegislationDatasetCreator(
                legislation_dir=str(self.legislation_dir),
                output_dir=str(self.datasets_dir),
                domain_focus="copyright"
            )
            
            # Create copyright-specific datasets
            success = dataset_creator.create_domain_datasets()
            
            if success:
                logger.info("ParaLlama copyright dataset creation completed successfully")
                return True
            else:
                logger.error("ParaLlama copyright dataset creation failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in ParaLlama copyright dataset creation: {e}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete Legal Llama copyright law pipeline"""
        logger.info("=== STARTING LEGAL LLAMA COPYRIGHT PIPELINE ===")
        start_time = time.time()
        
        try:
            # Phase 1: Download copyright legislation
            logger.info("Phase 1: Downloading copyright legislation...")
            self.controller.set_current_phase('legislation_download', {'step': 'copyright_docs'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            if not self.download_copyright_legislation():
                logger.error("Copyright legislation download failed, stopping pipeline")
                return False
            
            # Phase 2: Scrape copyright case law
            logger.info("Phase 2: Scraping copyright case law...")
            self.controller.set_current_phase('case_law_scraping', {'step': 'copyright_cases'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            if not self.scrape_copyright_case_law():
                logger.error("Copyright case law scraping failed, continuing with available data")
            
            # Phase 3: Generate Q&A pairs
            logger.info("Phase 3: Generating copyright Q&A pairs...")
            self.controller.set_current_phase('qa_generation', {'step': 'copyright_qa'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            if not self.generate_copyright_qa_pairs():
                logger.error("Copyright Q&A generation failed, continuing with available data")
            
            # Phase 4: Create final datasets
            logger.info("Phase 4: Creating copyright datasets...")
            self.controller.set_current_phase('dataset_creation', {'step': 'copyright_datasets'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            if not self.create_copyright_datasets():
                logger.error("Copyright dataset creation failed")
                return False
            
            # Calculate pipeline duration
            duration = time.time() - start_time
            logger.info(f"=== LEGAL LLAMA COPYRIGHT PIPELINE COMPLETED ===")
            logger.info(f"Total duration: {duration/60:.2f} minutes")
            
            # Generate summary report
            self.generate_pipeline_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Legal Llama copyright pipeline failed: {e}")
            return False
        finally:
            # Cleanup controller
            self.controller.cleanup()
    
    def generate_pipeline_summary(self):
        """Generate a summary report of the copyright pipeline results"""
        logger.info("=== LEGAL LLAMA COPYRIGHT PIPELINE SUMMARY ===")
        
        try:
            # Count legislation files
            legislation_count = 0
            if (self.legislation_dir / "text").exists():
                legislation_count = len(list((self.legislation_dir / "text").glob("*.txt")))
            
            # Count case law files
            case_law_count = 0
            if (self.case_law_dir / "text").exists():
                case_law_count = len(list((self.case_law_dir / "text").glob("*.txt")))
            
            # Count dataset files
            dataset_count = 0
            if self.datasets_dir.exists():
                dataset_count = len(list(self.datasets_dir.rglob("*.json")))
            
            logger.info(f"Copyright legislation items processed: {legislation_count}")
            logger.info(f"Copyright case law items processed: {case_law_count}")
            logger.info(f"Dataset files created: {dataset_count}")
            
            # Save summary to file
            summary = {
                "pipeline": "Legal Llama Copyright Law",
                "legislation_items": legislation_count,
                "case_law_items": case_law_count,
                "dataset_files": dataset_count,
                "directories": {
                    "legislation": str(self.legislation_dir),
                    "case_law": str(self.case_law_dir),
                    "datasets": str(self.datasets_dir)
                }
            }
            
            import json
            summary_file = self.datasets_dir / "copyright_pipeline_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Pipeline summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error generating pipeline summary: {e}")

def main():
    """Main function for Legal Llama Copyright Pipeline"""
    parser = argparse.ArgumentParser(
        description="Legal Llama Copyright Law Pipeline - Complete copyright law data collection and dataset creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python copyright_pipeline.py                                    # Run complete pipeline
  python copyright_pipeline.py --max-legislation 100             # Limit legislation items
  python copyright_pipeline.py --max-cases 50                    # Limit case law items
  python copyright_pipeline.py --output-dir ./copyright_data     # Custom output directory
        """
    )
    
    parser.add_argument(
        '--legislation-dir',
        default='generated/copyright_legislation',
        help='Directory for copyright legislation data'
    )
    
    parser.add_argument(
        '--case-law-dir', 
        default='generated/copyright_case_law',
        help='Directory for copyright case law data'
    )
    
    parser.add_argument(
        '--datasets-dir',
        default='generated/copyright_datasets', 
        help='Directory for final copyright datasets'
    )
    
    parser.add_argument(
        '--max-legislation',
        type=int,
        help='Maximum number of legislation items to download'
    )
    
    parser.add_argument(
        '--max-cases',
        type=int,
        help='Maximum number of case law items to download'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Base output directory (will create subdirectories within this)'
    )
    
    args = parser.parse_args()
    
    # Adjust directories if base output directory specified
    if args.output_dir:
        base_dir = Path(args.output_dir)
        args.legislation_dir = str(base_dir / "legislation")
        args.case_law_dir = str(base_dir / "case_law") 
        args.datasets_dir = str(base_dir / "datasets")
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    try:
        # Create and run the copyright pipeline
        pipeline = CopyrightLawPipeline(
            legislation_dir=args.legislation_dir,
            case_law_dir=args.case_law_dir,
            datasets_dir=args.datasets_dir,
            max_legislation_items=args.max_legislation,
            max_cases=args.max_cases
        )
        
        success = pipeline.run_complete_pipeline()
        
        if success:
            print("\n🎉 Legal Llama Copyright Pipeline completed successfully!")
            print(f"📂 Check your results in: {args.datasets_dir}")
        else:
            print("\n❌ Legal Llama Copyright Pipeline failed!")
            print("📋 Check logs/copyright_pipeline.log for details")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Legal Llama Copyright Pipeline interrupted by user")
        print("\n⏹️  Pipeline stopped by user")
    except Exception as e:
        logger.error(f"Legal Llama Copyright Pipeline error: {e}")
        print(f"\n💥 Pipeline error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()