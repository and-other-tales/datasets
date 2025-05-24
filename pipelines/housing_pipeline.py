#!/usr/bin/env python3
"""
Complete Housing Law Pipeline for Legal Llama Training

This script runs the entire process from downloading housing-specific UK legislation 
and case law to creating specialised datasets for housing law Legal Llama fine-tuning.
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

# Import our housing-specific modules  
from housing_legislation_downloader import HousingLegislationDownloader

# Conditional imports for modules that might require additional dependencies
try:
    from housing_bailii_scraper import HousingBailiiScraper
except ImportError:
    HousingBailiiScraper = None

try:
    from housing_QA_generator import HousingQAGenerator
except ImportError:
    HousingQAGenerator = None

try:
    from dataset_creator import UKLegislationDatasetCreator
except ImportError:
    UKLegislationDatasetCreator = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/housing_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LegalLlamaHousingLawPipeline:
    def __init__(
        self,
        legislation_dir: str = "housing_legislation",
        case_law_dir: str = "housing_case_law",
        dataset_dir: str = "housing_datasets",
        skip_legislation: bool = False,
        skip_case_law: bool = False,
        skip_qa_generation: bool = False,
        skip_dataset_creation: bool = False,
        max_cases: int = 500,
        enable_pause_controls: bool = True
    ):
        self.legislation_dir = legislation_dir
        self.case_law_dir = case_law_dir
        self.dataset_dir = dataset_dir
        self.skip_legislation = skip_legislation
        self.skip_case_law = skip_case_law
        self.skip_qa_generation = skip_qa_generation
        self.skip_dataset_creation = skip_dataset_creation
        self.max_cases = max_cases
        
        # Initialize pipeline controller
        self.controller = None
        if enable_pause_controls:
            try:
                self.controller = PipelineController()
                self.controller.register_callback('database_update', create_database_update_callback(self))
                self.controller.register_callback('dataset_creation', create_dataset_creation_callback(self))
            except Exception as e:
                logger.warning(f"Could not initialize pause controls: {e}")
                self.controller = None
        
        # Control message now handled by curses footer
        
        # Initialize components
        self.legislation_downloader = HousingLegislationDownloader(legislation_dir)
        self.case_law_scraper = HousingBailiiScraper(case_law_dir) if HousingBailiiScraper else None
        self.qa_generator = HousingQAGenerator() if HousingQAGenerator else None
        self.dataset_creator = UKLegislationDatasetCreator(
            source_dir=legislation_dir,
            output_dir=dataset_dir
        ) if UKLegislationDatasetCreator else None
    
    def run_legislation_download_phase(self) -> bool:
        """Download housing-specific legislation"""
        if self.skip_legislation:
            logger.info("Skipping legislation download phase as requested")
            return True
        
        logger.info("=== STARTING HOUSING LEGISLATION DOWNLOAD PHASE ===")
        
        # Check for pause/quit commands
        if self.controller:
            command = self.controller.check_for_commands()
            if command == 'quit':
                return False
            self.controller.wait_while_paused()
            self.controller.set_current_phase("Downloading housing legislation", {})
        
        try:
            # Load any previous progress
            self.legislation_downloader.load_progress()
            
            # Discover and download housing legislation
            logger.info("Discovering housing legislation...")
            if self.controller:
                command = self.controller.check_for_commands()
                if command == 'quit':
                    return False
                self.controller.wait_while_paused()
            
            self.legislation_downloader.run_housing_discovery()
            
            logger.info("Downloading housing legislation...")
            if self.controller:
                command = self.controller.check_for_commands()
                if command == 'quit':
                    return False
                self.controller.wait_while_paused()
            
            self.legislation_downloader.download_all_housing_legislation()
            
            # Generate summary
            summary = self.legislation_downloader.generate_housing_summary()
            
            logger.info("Housing legislation download phase completed!")
            logger.info(f"Downloaded {summary['total_downloaded']} housing acts")
            
            return True
            
        except Exception as e:
            logger.error(f"Housing legislation download phase failed: {e}")
            return False
    
    def run_case_law_scraping_phase(self) -> bool:
        """Scrape housing-specific case law"""
        if self.skip_case_law:
            logger.info("Skipping case law scraping phase as requested")
            return True
        
        if not self.case_law_scraper:
            logger.error("Case law scraper not available - missing dependencies")
            return False
        
        logger.info("=== STARTING HOUSING CASE LAW SCRAPING PHASE ===")
        
        # Check for pause/quit commands
        if self.controller:
            command = self.controller.check_for_commands()
            if command == 'quit':
                return False
            self.controller.wait_while_paused()
            self.controller.set_current_phase("Scraping housing case law", {})
        
        try:
            # Scrape housing cases
            logger.info(f"Scraping up to {self.max_cases} housing cases...")
            if self.controller:
                command = self.controller.check_for_commands()
                if command == 'quit':
                    return False
                self.controller.wait_while_paused()
            
            cases = self.case_law_scraper.scrape_all_housing_cases(self.max_cases)
            
            # Save cases
            if cases:
                self.case_law_scraper.save_housing_cases(cases)
                logger.info(f"Successfully scraped {len(cases)} housing cases")
            else:
                logger.warning("No housing cases were scraped")
            
            logger.info("Housing case law scraping phase completed!")
            return True
            
        except Exception as e:
            logger.error(f"Housing case law scraping phase failed: {e}")
            return False
    
    def run_qa_generation_phase(self) -> bool:
        """Generate housing-specific Q&A pairs"""
        if self.skip_qa_generation:
            logger.info("Skipping Q&A generation phase as requested")
            return True
        
        if not self.qa_generator:
            logger.error("Q&A generator not available - missing dependencies")
            return False
        
        logger.info("=== STARTING HOUSING Q&A GENERATION PHASE ===")
        
        # Check for pause/quit commands
        if self.controller:
            command = self.controller.check_for_commands()
            if command == 'quit':
                return False
            self.controller.wait_while_paused()
            self.controller.set_current_phase("Generating housing Q&A pairs", {})
        
        try:
            # Check for source files
            legislation_path = Path(self.legislation_dir)
            case_law_file = Path(self.case_law_dir) / "housing_cases.json"
            
            if not legislation_path.exists():
                logger.error(f"Legislation directory {legislation_path} does not exist")
                return False
            
            # Generate Q&A pairs
            if self.controller:
                command = self.controller.check_for_commands()
                if command == 'quit':
                    return False
                self.controller.wait_while_paused()
            
            qa_output_file = Path(self.dataset_dir) / "housing_qa_dataset.json"
            qa_pairs = self.qa_generator.process_all_housing_sources(
                self.legislation_dir,
                str(case_law_file) if case_law_file.exists() else None,
                str(qa_output_file)
            )
            
            if qa_pairs:
                logger.info(f"Generated {len(qa_pairs)} housing Q&A pairs")
            else:
                logger.warning("No Q&A pairs were generated")
            
            logger.info("Housing Q&A generation phase completed!")
            return True
            
        except Exception as e:
            logger.error(f"Housing Q&A generation phase failed: {e}")
            return False
    
    def run_dataset_creation_phase(self) -> bool:
        """Create housing datasets for training"""
        if self.skip_dataset_creation:
            logger.info("Skipping dataset creation phase as requested")
            return True
        
        if not self.dataset_creator:
            logger.error("Dataset creator not available - missing dependencies")
            return False
        
        logger.info("=== STARTING HOUSING DATASET CREATION PHASE ===")
        
        # Check for pause/quit commands
        if self.controller:
            command = self.controller.check_for_commands()
            if command == 'quit':
                return False
            self.controller.wait_while_paused()
            self.controller.set_current_phase("Creating housing datasets", {})
        
        try:
            # Check if source data exists
            source_path = Path(self.legislation_dir)
            if not source_path.exists():
                logger.error(f"Source directory {source_path} does not exist")
                return False
            
            # Create datasets
            logger.info("Creating housing datasets...")
            if self.controller:
                command = self.controller.check_for_commands()
                if command == 'quit':
                    return False
                self.controller.wait_while_paused()
            
            datasets = self.dataset_creator.create_all_datasets()
            
            logger.info("Housing dataset creation phase completed!")
            
            # Print summary
            total_examples = sum(len(dataset) for dataset in datasets.values())
            logger.info(f"Created {len(datasets)} dataset splits with {total_examples} total examples")
            
            return True
            
        except Exception as e:
            logger.error(f"Housing dataset creation phase failed: {e}")
            return False
    
    def run_complete_housing_pipeline(self) -> bool:
        """Run the complete housing law pipeline"""
        logger.info("=== STARTING COMPLETE LEGAL LLAMA HOUSING LAW PIPELINE ===")
        start_time = time.time()
        
        try:
            # Phase 1: Download housing legislation
            if not self.run_legislation_download_phase():
                logger.error("Housing legislation download phase failed")
                return False
            
            # Phase 2: Scrape housing case law
            if not self.run_case_law_scraping_phase():
                logger.error("Housing case law scraping phase failed")
                return False
            
            # Phase 3: Generate housing Q&A pairs
            if not self.run_qa_generation_phase():
                logger.error("Housing Q&A generation phase failed")
                return False
            
            # Phase 4: Create datasets
            if not self.run_dataset_creation_phase():
                logger.error("Housing dataset creation phase failed")
                return False
            
            # Success
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("=== LEGAL LLAMA HOUSING LAW PIPELINE COMPLETED SUCCESSFULLY ===")
            logger.info(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            
            # Final summary
            self.print_final_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Complete housing pipeline failed: {e}")
            return False
        finally:
            # Cleanup controller
            if self.controller:
                self.controller.cleanup()
    
    def print_final_summary(self):
        """Print final summary of housing pipeline results"""
        print(f"\n{'='*70}")
        print("LEGAL LLAMA HOUSING LAW PIPELINE COMPLETION SUMMARY")
        print(f"{'='*70}")
        
        # Legislation summary
        legislation_path = Path(self.legislation_dir)
        if legislation_path.exists():
            text_files = len(list((legislation_path / "text").glob("*.txt"))) if (legislation_path / "text").exists() else 0
            xml_files = len(list((legislation_path / "xml").glob("*.xml"))) if (legislation_path / "xml").exists() else 0
            
            print(f"\nHOUSING LEGISLATION DOWNLOADED:")
            print(f"  Text files: {text_files}")
            print(f"  XML files: {xml_files}")
            print(f"  Location: {legislation_path.absolute()}")
        
        # Case law summary
        case_law_path = Path(self.case_law_dir)
        if case_law_path.exists():
            case_file = case_law_path / "housing_cases.json"
            if case_file.exists():
                try:
                    import json
                    with open(case_file, 'r') as f:
                        cases = json.load(f)
                    print(f"\nHOUSING CASE LAW SCRAPED:")
                    print(f"  Cases: {len(cases)}")
                    print(f"  Location: {case_file.absolute()}")
                except:
                    print(f"\nHOUSING CASE LAW: File exists at {case_file}")
        
        # Q&A pairs summary
        qa_file = Path(self.dataset_dir) / "housing_qa_dataset.json"
        if qa_file.exists():
            try:
                import json
                with open(qa_file, 'r') as f:
                    qa_data = json.load(f)
                print(f"\nHOUSING Q&A PAIRS GENERATED:")
                print(f"  Q&A pairs: {len(qa_data)}")
                print(f"  Location: {qa_file.absolute()}")
            except:
                print(f"\nHOUSING Q&A PAIRS: File exists at {qa_file}")
        
        # Dataset summary
        dataset_path = Path(self.dataset_dir)
        if dataset_path.exists():
            print(f"\nHOUSING DATASETS CREATED:")
            
            # Check for final datasets
            final_dir = dataset_path / "final"
            if final_dir.exists():
                parquet_files = list(final_dir.glob("*.parquet"))
                print(f"  Dataset splits: {len(parquet_files)}")
                for pf in parquet_files:
                    print(f"    - {pf.stem}")
            
            print(f"  Location: {dataset_path.absolute()}")
        
        print(f"\nHOUSING LAW SPECIALIZATION:")
        print(f"  ✓ Housing Acts (1985, 1988, 1996, 2004)")
        print(f"  ✓ Tenancy law and eviction procedures")
        print(f"  ✓ Landlord and tenant rights/obligations")
        print(f"  ✓ Leasehold and property law")
        print(f"  ✓ Housing case law and precedents")
        
        print(f"\nUSAGE:")
        print(f"  Training: Load datasets from {dataset_path}/final/")
        print(f"  Q&A Training: Use housing_qa_dataset.json for supervised fine-tuning")
        print(f"  Research: Individual files for specific housing law topics")
        
        print(f"\nNEXT STEPS:")
        print(f"  1. Review housing_qa_summary.json for specialised statistics")
        print(f"  2. Fine-tune your model on housing-specific Q&A pairs")
        print(f"  3. Use for housing law chatbot or legal assistant development")
        print(f"  4. Combine with general legal training for comprehensive coverage")
        
        print(f"{'='*70}\n")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Complete Housing Law Pipeline for Legal Llama Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python housing_pipeline.py                           # Run complete housing pipeline
  python housing_pipeline.py --skip-legislation       # Only scrape cases and generate Q&A
  python housing_pipeline.py --skip-case-law          # Only download legislation and generate Q&A
  python housing_pipeline.py --max-cases 1000         # Scrape up to 1000 housing cases
  python housing_pipeline.py --legislation-dir ./housing_acts --case-law-dir ./housing_cases
        """
    )
    
    parser.add_argument(
        '--legislation-dir',
        default='housing_legislation',
        help='Directory to store housing legislation files (default: housing_legislation)'
    )
    
    parser.add_argument(
        '--case-law-dir',
        default='housing_case_law',
        help='Directory to store housing case law (default: housing_case_law)'
    )
    
    parser.add_argument(
        '--dataset-dir',
        default='housing_datasets',
        help='Directory to store created datasets (default: housing_datasets)'
    )
    
    parser.add_argument(
        '--skip-legislation',
        action='store_true',
        help='Skip housing legislation download phase'
    )
    
    parser.add_argument(
        '--skip-case-law',
        action='store_true',
        help='Skip housing case law scraping phase'
    )
    
    parser.add_argument(
        '--skip-qa',
        action='store_true',
        help='Skip housing Q&A generation phase'
    )
    
    parser.add_argument(
        '--skip-datasets',
        action='store_true',
        help='Skip dataset creation phase'
    )
    
    parser.add_argument(
        '--max-cases',
        type=int,
        default=500,
        help='Maximum number of housing cases to scrape (default: 500)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_legislation and args.skip_case_law and args.skip_qa and args.skip_datasets:
        logger.error("Cannot skip all phases")
        sys.exit(1)
    
    # Create pipeline
    pipeline = LegalLlamaHousingLawPipeline(
        legislation_dir=args.legislation_dir,
        case_law_dir=args.case_law_dir,
        dataset_dir=args.dataset_dir,
        skip_legislation=args.skip_legislation,
        skip_case_law=args.skip_case_law,
        skip_qa_generation=args.skip_qa,
        skip_dataset_creation=args.skip_datasets,
        max_cases=args.max_cases
    )
    
    # Run pipeline
    try:
        success = pipeline.run_complete_housing_pipeline()
        
        if success:
            logger.info("Housing law pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("Housing law pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Housing law pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Housing law pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()