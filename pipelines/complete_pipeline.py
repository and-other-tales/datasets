#!/usr/bin/env python3
"""
Complete UK Legislation Pipeline for Legal Llama Training

This script runs the entire process from downloading UK legislation
to creating complete datasets suitable for Legal Llama fine-tuning.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

# Import our custom modules
from improved_downloader import ImprovedUKLegislationDownloader
from dataset_creator import UKLegislationDatasetCreator
from QA_pairs import UKLegislationQAGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LegalLlamaUKLegislationPipeline:
    def __init__(
        self,
        output_dir: str = "uk_legislation",
        dataset_dir: str = "uk_legislation_datasets",
        skip_download: bool = False,
        skip_dataset_creation: bool = False,
        skip_qa_generation: bool = False,
        tokenizer_name: str = "microsoft/DialoGPT-medium"
    ):
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir
        self.skip_download = skip_download
        self.skip_dataset_creation = skip_dataset_creation
        self.skip_qa_generation = skip_qa_generation
        self.tokenizer_name = tokenizer_name
        
        # Initialize components
        self.downloader = ImprovedUKLegislationDownloader(output_dir)
        self.dataset_creator = UKLegislationDatasetCreator(
            source_dir=output_dir,
            output_dir=dataset_dir,
            tokenizer_name=tokenizer_name
        )
        self.qa_generator = UKLegislationQAGenerator()
    
    def run_download_phase(self) -> bool:
        """Run the legislation download phase"""
        if self.skip_download:
            logger.info("Skipping download phase as requested")
            return True
        
        logger.info("=== STARTING DOWNLOAD PHASE ===")
        
        try:
            # Load any previous progress
            self.downloader.load_progress()
            
            # Discover all legislation
            logger.info("Discovering legislation...")
            self.downloader.discover_legislation_systematically()
            
            # Download all legislation
            logger.info("Downloading legislation...")
            self.downloader.download_all_legislation()
            
            # Verify downloads
            logger.info("Verifying downloads...")
            stats = self.downloader.verify_downloads()
            
            logger.info("Download phase completed successfully!")
            logger.info(f"Download statistics: {stats}")
            
            # Check if we have sufficient data
            total_files = stats.get('xml_files', 0) + stats.get('text_files', 0)
            if total_files < 10:
                logger.warning(f"Only {total_files} files downloaded. Dataset may be incomplete.")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Download phase failed: {e}")
            return False
    
    def run_dataset_creation_phase(self) -> bool:
        """Run the dataset creation phase"""
        if self.skip_dataset_creation:
            logger.info("Skipping dataset creation phase as requested")
            return True
        
        logger.info("=== STARTING DATASET CREATION PHASE ===")
        
        try:
            # Check if source data exists
            source_path = Path(self.output_dir)
            if not source_path.exists():
                logger.error(f"Source directory {source_path} does not exist")
                return False
            
            # Check for text or XML files
            text_files = list((source_path / "text").glob("*.txt")) if (source_path / "text").exists() else []
            xml_files = list((source_path / "xml").glob("*.xml")) if (source_path / "xml").exists() else []
            
            if not text_files and not xml_files:
                logger.error("No text or XML files found for dataset creation")
                return False
            
            logger.info(f"Found {len(text_files)} text files and {len(xml_files)} XML files")
            
            # Create datasets
            logger.info("Creating comprehensive datasets...")
            datasets = self.dataset_creator.create_all_datasets()
            
            logger.info("Dataset creation phase completed successfully!")
            
            # Print summary
            total_examples = sum(len(dataset) for dataset in datasets.values())
            logger.info(f"Created {len(datasets)} dataset splits with {total_examples} total examples")
            
            return True
            
        except Exception as e:
            logger.error(f"Dataset creation phase failed: {e}")
            return False
    
    def run_qa_generation_phase(self) -> bool:
        """Run the Q&A generation phase"""
        if self.skip_qa_generation:
            logger.info("Skipping Q&A generation phase as requested")
            return True
        
        logger.info("=== STARTING Q&A GENERATION PHASE ===")
        
        try:
            # Check if source data exists
            source_path = Path(self.output_dir)
            if not source_path.exists():
                logger.error(f"Source directory {source_path} does not exist")
                return False
            
            # Check for text files
            text_files = []
            if (source_path / "text").exists():
                text_files.extend(list((source_path / "text").glob("*.txt")))
            text_files.extend(list(source_path.glob("*.txt")))
            
            if not text_files:
                logger.error("No text files found for Q&A generation")
                return False
            
            logger.info(f"Found {len(text_files)} text files for Q&A generation")
            
            # Generate Q&A pairs
            qa_output_file = Path(self.dataset_dir) / "qa_pairs_dataset.json"
            qa_pairs = self.qa_generator.process_all_legislation(
                self.output_dir, 
                str(qa_output_file)
            )
            
            if not qa_pairs:
                logger.warning("No Q&A pairs were generated")
                return False
            
            logger.info("Q&A generation phase completed successfully!")
            logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
            
            return True
            
        except Exception as e:
            logger.error(f"Q&A generation phase failed: {e}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete pipeline from download to dataset creation"""
        logger.info("=== STARTING COMPLETE LEGAL LLAMA UK LEGISLATION PIPELINE ===")
        start_time = time.time()
        
        try:
            # Phase 1: Download legislation
            if not self.run_download_phase():
                logger.error("Download phase failed")
                return False
            
            # Phase 2: Create datasets
            if not self.run_dataset_creation_phase():
                logger.error("Dataset creation phase failed")
                return False
            
            # Phase 3: Generate Q&A pairs
            if not self.run_qa_generation_phase():
                logger.error("Q&A generation phase failed")
                return False
            
            # Success
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("=== LEGAL LLAMA PIPELINE COMPLETED SUCCESSFULLY ===")
            logger.info(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            
            # Final summary
            self.print_final_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Complete pipeline failed: {e}")
            return False
    
    def print_final_summary(self):
        """Print final summary of pipeline results"""
        print(f"\n{'='*60}")
        print("LEGAL LLAMA UK LEGISLATION PIPELINE COMPLETION SUMMARY")
        print(f"{'='*60}")
        
        # Download summary
        output_path = Path(self.output_dir)
        if output_path.exists():
            text_files = len(list((output_path / "text").glob("*.txt"))) if (output_path / "text").exists() else 0
            xml_files = len(list((output_path / "xml").glob("*.xml"))) if (output_path / "xml").exists() else 0
            html_files = len(list((output_path / "html").glob("*.html"))) if (output_path / "html").exists() else 0
            metadata_files = len(list((output_path / "metadata").glob("*.json"))) if (output_path / "metadata").exists() else 0
            
            print(f"\nDOWNLOADED FILES:")
            print(f"  Text files: {text_files}")
            print(f"  XML files: {xml_files}")
            print(f"  HTML files: {html_files}")
            print(f"  Metadata files: {metadata_files}")
            print(f"  Location: {output_path.absolute()}")
        
        # Dataset summary
        dataset_path = Path(self.dataset_dir)
        if dataset_path.exists():
            print(f"\nCREATED DATASETS:")
            
            # Check for final datasets
            final_dir = dataset_path / "final"
            if final_dir.exists():
                parquet_files = list(final_dir.glob("*.parquet"))
                print(f"  Dataset splits: {len(parquet_files)}")
                for pf in parquet_files:
                    print(f"    - {pf.stem}")
            
            # Check for Q&A pairs
            qa_file = dataset_path / "qa_pairs_dataset.json"
            if qa_file.exists():
                try:
                    import json
                    with open(qa_file, 'r') as f:
                        qa_data = json.load(f)
                    print(f"  Q&A pairs: {len(qa_data)} pairs generated")
                except:
                    print(f"  Q&A pairs: Dataset file exists")
            
            print(f"  Location: {dataset_path.absolute()}")
            
            # Available formats
            formats = []
            if (dataset_path / "final").exists():
                formats.append("HuggingFace Dataset")
            if list(dataset_path.glob("**/*.parquet")):
                formats.append("Parquet")
            if list(dataset_path.glob("**/*.json")):
                formats.append("JSON")
            
            print(f"  Available formats: {', '.join(formats)}")
        
        print(f"\nUSAGE:")
        print(f"  Training: Load datasets from {dataset_path}/final/")
        print(f"  Research: Use individual splits for specific tasks")
        print(f"  Integration: Compatible with HuggingFace Transformers")
        
        print(f"\nNEXT STEPS:")
        print(f"  1. Review validation_report.json for dataset statistics")
        print(f"  2. Load datasets using: load_from_disk('{dataset_path}/final/uk_legislation_complete')")
        print(f"  3. Use Q&A pairs from qa_pairs_dataset.json for supervised fine-tuning")
        print(f"  4. Fine-tune your Legal Llama model with the created datasets")
        
        print(f"{'='*60}\n")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Complete UK Legislation Pipeline for Legal Llama Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python complete_pipeline.py                    # Run complete pipeline
  python complete_pipeline.py --skip-download   # Only create datasets from existing files
  python complete_pipeline.py --skip-datasets   # Only download legislation
  python complete_pipeline.py --output-dir ./my_legislation --dataset-dir ./my_datasets
        """
    )
    
    parser.add_argument(
        '--output-dir',
        default='uk_legislation',
        help='Directory to store downloaded legislation files (default: uk_legislation)'
    )
    
    parser.add_argument(
        '--dataset-dir',
        default='uk_legislation_datasets',
        help='Directory to store created datasets (default: uk_legislation_datasets)'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip the download phase and only create datasets from existing files'
    )
    
    parser.add_argument(
        '--skip-datasets',
        action='store_true',
        help='Skip dataset creation phase and only download legislation'
    )
    
    parser.add_argument(
        '--skip-qa',
        action='store_true',
        help='Skip Q&A generation phase'
    )
    
    parser.add_argument(
        '--tokenizer',
        default='microsoft/DialoGPT-medium',
        help='Tokenizer to use for text processing (default: microsoft/DialoGPT-medium)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_download and args.skip_datasets and args.skip_qa:
        logger.error("Cannot skip all phases")
        sys.exit(1)
    
    # Create pipeline
    pipeline = LegalLlamaUKLegislationPipeline(
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        skip_download=args.skip_download,
        skip_dataset_creation=args.skip_datasets,
        skip_qa_generation=args.skip_qa,
        tokenizer_name=args.tokenizer
    )
    
    # Run pipeline
    try:
        success = pipeline.run_complete_pipeline()
        
        if success:
            logger.info("Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()