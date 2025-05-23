#!/usr/bin/env python3
"""
Legal Llama - Main Entry Point

This script provides a unified interface to run various legal data collection pipelines
and enhanced dataset generation for training domain-specialist Legal Llama models.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_dynamic_pipeline(args):
    """Run othertales Dynamic Pipeline for any URL"""
    from pipelines.dynamic_pipeline import DynamicDatasetPipeline
    
    try:
        # Get URL from user if not provided
        if not hasattr(args, 'url') or not args.url:
            url = input("Enter the URL to create datasets from: ").strip()
            if not url:
                print("URL is required for dynamic pipeline")
                return
        else:
            url = args.url
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        output_dir = args.output_dir or 'generated/dynamic_datasets'
        
        print(f"🚀 Starting othertales Dynamic Pipeline for: {url}")
        
        pipeline = DynamicDatasetPipeline(url, output_dir)
        result = pipeline.run_dynamic_pipeline()
        
        if result.get("status") == "success":
            print(f"\n🎉 othertales Dynamic Pipeline completed successfully!")
            print(f"📊 Generated {result['total_examples']} training examples")
            print(f"🎯 Domain: {result['domain']} - {result['specialization']}")
            print(f"📂 Results: {result['output_directory']}")
        else:
            print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error running dynamic pipeline: {e}")

def run_hmrc_scraper(args):
    """Run ParaLlama HMRC tax documentation scraper"""
    from pipelines.hmrc_scraper import main as hmrc_main
    
    # Ensure output directory exists
    output_dir = args.output_dir or 'generated/hmrc_documentation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up arguments for hmrc_scraper
    hmrc_args = [
        '--output-dir', output_dir
    ]
    
    if args.max_documents:
        hmrc_args.extend(['--max-documents', str(args.max_documents)])
    
    if args.discover_only:
        hmrc_args.append('--discover-only')
    
    # Override sys.argv for the hmrc_scraper
    original_argv = sys.argv
    sys.argv = ['hmrc_scraper.py'] + hmrc_args
    
    try:
        hmrc_main()
    finally:
        sys.argv = original_argv

def run_housing_pipeline(args):
    """Run housing legislation and case law pipeline"""
    from pipelines.housing_pipeline import main as housing_main
    
    # Set up arguments for housing pipeline
    housing_args = []
    
    if args.output_dir:
        housing_args.extend(['--output-dir', args.output_dir])
    
    if args.max_documents:
        housing_args.extend(['--max-documents', str(args.max_documents)])
    
    # Override sys.argv for the housing pipeline
    original_argv = sys.argv
    sys.argv = ['housing_pipeline.py'] + housing_args
    
    try:
        housing_main()
    except ImportError:
        print("Housing pipeline main function not found. Running housing pipeline directly...")
        import pipelines.housing_pipeline
    finally:
        sys.argv = original_argv

def run_bailii_scraper(args):
    """Run BAILII case law scraper"""
    from pipelines.bailii_scraper import main as bailii_main
    
    # Set up arguments for BAILII scraper
    bailii_args = []
    
    if args.output_dir:
        bailii_args.extend(['--output-dir', args.output_dir])
    
    if args.max_documents:
        bailii_args.extend(['--max-documents', str(args.max_documents)])
    
    # Override sys.argv for the BAILII scraper
    original_argv = sys.argv
    sys.argv = ['bailii_scraper.py'] + bailii_args
    
    try:
        bailii_main()
    except ImportError:
        print("BAILII scraper main function not found. Running BAILII scraper directly...")
        import pipelines.bailii_scraper
    finally:
        sys.argv = original_argv

def run_complete_pipeline(args):
    """Run the complete data collection pipeline"""
    from pipelines.complete_pipeline import main as complete_main
    
    # Set up arguments for complete pipeline
    complete_args = []
    
    if args.output_dir:
        complete_args.extend(['--output-dir', args.output_dir])
    
    if args.max_documents:
        complete_args.extend(['--max-documents', str(args.max_documents)])
    
    # Override sys.argv for the complete pipeline
    original_argv = sys.argv
    sys.argv = ['complete_pipeline.py'] + complete_args
    
    try:
        complete_main()
    except ImportError:
        print("Complete pipeline main function not found. Running complete pipeline directly...")
        import pipelines.complete_pipeline
    finally:
        sys.argv = original_argv

def run_qa_generator(args):
    """Run Q&A pair generator"""
    from pipelines.housing_QA_generator import main as qa_main
    
    # Set up arguments for QA generator
    qa_args = []
    
    if args.input_dir:
        qa_args.extend(['--input-dir', args.input_dir])
    
    if args.output_dir:
        qa_args.extend(['--output-dir', args.output_dir])
    
    # Override sys.argv for the QA generator
    original_argv = sys.argv
    sys.argv = ['housing_QA_generator.py'] + qa_args
    
    try:
        qa_main()
    except ImportError:
        print("QA generator main function not found. Running QA generator directly...")
        import pipelines.housing_QA_generator
    finally:
        sys.argv = original_argv

def run_database_ingestion(args):
    """Run ParaLlama database ingestion utility"""
    from utils.multi_database_ingestion import main as db_main
    
    # Set up arguments for database ingestion
    db_args = []
    
    if args.input_dir:
        db_args.extend(['--input-dir', args.input_dir])
    
    # Override sys.argv for the database ingestion
    original_argv = sys.argv
    sys.argv = ['multi_database_ingestion.py'] + db_args
    
    try:
        db_main()
    except ImportError:
        print("ParaLlama database ingestion main function not found. Running database ingestion directly...")
        import utils.multi_database_ingestion
    finally:
        sys.argv = original_argv

def run_copyright_pipeline(args):
    """Run Legal Llama copyright law pipeline"""
    from pipelines.copyright_pipeline import main as copyright_main
    
    # Set up arguments for copyright pipeline
    copyright_args = []
    
    if args.output_dir:
        copyright_args.extend(['--output-dir', args.output_dir])
    
    if args.max_documents:
        copyright_args.extend(['--max-documents', str(args.max_documents)])
    
    # Override sys.argv for the copyright pipeline
    original_argv = sys.argv
    sys.argv = ['copyright_pipeline.py'] + copyright_args
    
    try:
        copyright_main()
    except ImportError:
        print("Legal Llama copyright pipeline main function not found. Running copyright pipeline directly...")
        import pipelines.copyright_pipeline
    finally:
        sys.argv = original_argv

def run_legal_reasoning_enhancer(args):
    """Run legal reasoning dataset enhancer"""
    from pipelines.legal_reasoning_enhancer import main as enhancer_main
    
    # Set up arguments
    enhancer_args = []
    
    if args.input_dir:
        enhancer_args.extend(['--input-dir', args.input_dir])
    
    if args.output_dir:
        enhancer_args.extend(['--output-dir', args.output_dir])
    
    # Override sys.argv
    original_argv = sys.argv
    sys.argv = ['legal_reasoning_enhancer.py'] + enhancer_args
    
    try:
        enhancer_main()
    finally:
        sys.argv = original_argv

def run_tax_scenario_generator(args):
    """Run tax scenario generator"""
    from pipelines.tax_scenario_generator import main as tax_main
    
    # Set up arguments
    tax_args = []
    
    if args.input_dir:
        tax_args.extend(['--input-dir', args.input_dir])
    
    if args.output_dir:
        tax_args.extend(['--output-dir', args.output_dir])
    
    # Override sys.argv
    original_argv = sys.argv
    sys.argv = ['tax_scenario_generator.py'] + tax_args
    
    try:
        tax_main()
    finally:
        sys.argv = original_argv

def run_advanced_qa_generator(args):
    """Run advanced Q&A generator"""
    from pipelines.advanced_qa_generator import main as qa_main
    
    # Set up arguments
    qa_args = []
    
    if args.input_dir:
        qa_args.extend(['--input-dir', args.input_dir])
    
    if args.output_dir:
        qa_args.extend(['--output-dir', args.output_dir])
    
    # Override sys.argv
    original_argv = sys.argv
    sys.argv = ['advanced_qa_generator.py'] + qa_args
    
    try:
        qa_main()
    finally:
        sys.argv = original_argv

def run_llama_training_optimizer(args):
    """Run Legal Llama training dataset optimizer"""
    from utils.llama_training_optimizer import main as optimizer_main
    
    # Set up arguments
    optimizer_args = []
    
    if args.input_dir:
        optimizer_args.extend(['--input-dir', args.input_dir])
    
    if args.output_dir:
        optimizer_args.extend(['--output-dir', args.output_dir])
    
    # Override sys.argv
    original_argv = sys.argv
    sys.argv = ['llama_training_optimizer.py'] + optimizer_args
    
    try:
        optimizer_main()
    finally:
        sys.argv = original_argv

def run_enhanced_complete_pipeline(args):
    """Run complete enhanced pipeline for Legal Llama training"""
    print("=== Running Enhanced Complete Pipeline for Legal Llama Training ===")
    
    # Step 1: Collect base data
    print("\n1. Collecting HMRC tax documentation...")
    run_hmrc_scraper(args)
    
    print("\n2. Collecting housing legislation and case law...")
    run_housing_pipeline(args)
    
    print("\n3. Collecting additional case law from BAILII...")
    run_bailii_scraper(args)
    
    # Step 2: Generate enhanced datasets
    print("\n4. Enhancing legal reasoning datasets...")
    run_legal_reasoning_enhancer(args)
    
    print("\n5. Generating tax scenarios...")
    run_tax_scenario_generator(args)
    
    print("\n6. Creating advanced Q&A pairs...")
    run_advanced_qa_generator(args)
    
    # Step 3: Optimize for Legal Llama training
    print("\n7. Optimizing datasets for Legal Llama 3.1 training...")
    run_llama_training_optimizer(args)
    
    print("\n=== Enhanced Complete Pipeline Complete ===")
    print("Your datasets are now ready for training domain-specialist Legal Llama models!")

def show_interactive_menu():
    """Show interactive menu for pipeline selection"""
    while True:
        print("\n" + "="*60)
        print("              othertales Datasets Tools")
        print("="*60)
        print()
        print("DATA COLLECTION PIPELINES:")
        print("  1. othertales Dynamic Pipeline (Any URL)")
        print("  2. HMRC Tax Documentation Scraper")
        print("  3. Housing Legislation & Case Law Pipeline") 
        print("  4. BAILII Case Law Scraper")
        print("  5. Copyright Law Pipeline")
        print("  6. Complete Data Collection Pipeline")
        print()
        print("DATASET ENHANCEMENT (for LLM Training):")
        print("  7. Legal Reasoning Enhancer")
        print("  8. Tax Scenario Generator")
        print("  9. Advanced Q&A Generator")
        print(" 10. Legal Llama Training Optimiser")
        print()
        print("COMPLETE WORKFLOWS:")
        print(" 11. Enhanced Complete Pipeline (All Steps)")
        print(" 12. Q&A Generation Only")
        print(" 13. Database Ingestion")
        print()
        print("OTHER OPTIONS:")
        print(" 14. Show Pipeline Status")
        print(" 15. View Documentation")
        print("  0. Exit")
        print()
        
        try:
            choice = input("Select an option (0-15): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                _run_with_menu_args(run_dynamic_pipeline, "othertales Dynamic Pipeline")
            elif choice == "2":
                _run_with_menu_args(run_hmrc_scraper, "HMRC Scraper")
            elif choice == "3":
                _run_with_menu_args(run_housing_pipeline, "Housing Pipeline")
            elif choice == "4":
                _run_with_menu_args(run_bailii_scraper, "BAILII Scraper")
            elif choice == "5":
                _run_with_menu_args(run_copyright_pipeline, "Copyright Pipeline")
            elif choice == "6":
                _run_with_menu_args(run_complete_pipeline, "Complete Pipeline")
            elif choice == "7":
                _run_with_menu_args(run_legal_reasoning_enhancer, "Legal Reasoning Enhancer")
            elif choice == "8":
                _run_with_menu_args(run_tax_scenario_generator, "Tax Scenario Generator")
            elif choice == "9":
                _run_with_menu_args(run_advanced_qa_generator, "Advanced Q&A Generator")
            elif choice == "10":
                _run_with_menu_args(run_llama_training_optimizer, "Legal Llama Training Optimiser")
            elif choice == "11":
                _run_with_menu_args(run_enhanced_complete_pipeline, "Enhanced Complete Pipeline")
            elif choice == "12":
                _run_with_menu_args(run_qa_generator, "Q&A Generator")
            elif choice == "13":
                _run_with_menu_args(run_database_ingestion, "Database Ingestion")
            elif choice == "14":
                _show_pipeline_status()
            elif choice == "15":
                _show_documentation()
            else:
                print("Invalid choice. Please select a number between 0-15.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def _run_with_menu_args(pipeline_func, pipeline_name):
    """Run a pipeline with interactive argument collection"""
    print(f"\n=== Running {pipeline_name} ===")
    
    # Create args object with common options
    class Args:
        def __init__(self):
            self.input_dir = None
            self.output_dir = None
            self.max_documents = None
            self.discover_only = False
            self.url = None
    
    args = Args()
    
    # Special handling for Dynamic Pipeline
    if pipeline_name == "othertales Dynamic Pipeline":
        url = input("Enter URL to create datasets from: ").strip()
        if not url:
            print("URL is required for dynamic pipeline")
            return
        args.url = url
        
        output_dir = input("Output directory (press Enter for default): ").strip()
        if output_dir:
            args.output_dir = output_dir
    else:
        # Get common arguments interactively
        if pipeline_name in ["HMRC Scraper", "Housing Pipeline", "BAILII Scraper", "Complete Pipeline", "Copyright Pipeline"]:
            max_docs = input("Maximum documents to download (press Enter for all): ").strip()
            if max_docs:
                try:
                    args.max_documents = int(max_docs)
                except ValueError:
                    print("Invalid number, using default")
            
            if pipeline_name == "HMRC Scraper":
                discover = input("Discovery only? (y/N): ").strip().lower()
                args.discover_only = discover == 'y'
        
        output_dir = input(f"Output directory (press Enter for default): ").strip()
        if output_dir:
            args.output_dir = output_dir
        
        input_dir = input(f"Input directory (press Enter for default): ").strip()
        if input_dir:
            args.input_dir = input_dir
    
    try:
        pipeline_func(args)
        print(f"\n{pipeline_name} completed successfully!")
    except Exception as e:
        print(f"\nError running {pipeline_name}: {e}")
    
    input("\nPress Enter to continue...")

def _show_pipeline_status():
    """Show status of data directories"""
    print("\n=== Pipeline Status ===")
    
    generated_dir = Path("generated")
    if not generated_dir.exists():
        print("No generated data found.")
        return
    
    for subdir in generated_dir.iterdir():
        if subdir.is_dir():
            file_count = len(list(subdir.rglob("*")))
            print(f"{subdir.name}: {file_count} files")
    
    input("\nPress Enter to continue...")

def _show_documentation():
    """Show quick documentation"""
    print("""
=== LEGAL LLAMA DATASETS DOCUMENTATION ===

PURPOSE:
Train domain-specialist Legal Llama models for UK legal and tax expertise.

RECOMMENDED WORKFLOW:
1. Run Enhanced Complete Pipeline (#9) for full data collection and enhancement
2. Use Legal Llama Training Optimiser (#8) output with HuggingFace AutoTrain Advanced
3. Train separate Legal Llama models for legal and tax specialisation

KEY FEATURES:
- UK Government Content API integration for reliable data extraction
- Multi-step reasoning enhancement for complex legal analysis
- Tax calculation and optimisation scenario generation
- Adversarial training data for robust argument handling
- Progressive training phases for building expertise

TARGET MODELS:
- Legal Specialist: Counter arguments, provide legal analysis
- Tax Specialist: Ensure compliance, maximise legitimate savings

For detailed documentation, see README.md
""")
    input("\nPress Enter to continue...")

def main():
    """Main entry point"""
    # Check if running in interactive mode (no command line arguments)
    if len(sys.argv) == 1:
        show_interactive_menu()
        return
    
    parser = argparse.ArgumentParser(
        description="Legal Llama Datasets - Unified Legal Data Collection and Enhancement Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available pipelines:
  hmrc                    - Scrape HMRC tax documentation from gov.uk (with Content API)
  housing                 - Collect housing legislation and case law
  bailii                  - Scrape case law from BAILII
  complete                - Run complete data collection pipeline
  enhanced-complete       - Run enhanced complete pipeline for LLM training
  qa-generator            - Generate Q&A pairs from collected data
  advanced-qa             - Generate advanced multi-step Q&A for LLM training
  legal-enhancer          - Enhance legal datasets with reasoning patterns
  tax-scenarios           - Generate tax calculation and optimization scenarios
  llama-optimizer         - Optimize datasets for Legal Llama 3.1 training
  db-ingestion            - Ingest data into databases
  menu                    - Show interactive menu

Examples:
  python main.py                                    # Show interactive menu
  python main.py menu                               # Show interactive menu
  python main.py enhanced-complete                  # Run full enhanced pipeline
  python main.py hmrc --max-documents 100          # Collect HMRC data
  python main.py llama-optimizer                   # Prepare for Legal Llama training
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='pipeline', help='Pipeline to run')
    
    # Data collection pipelines
    dynamic_parser = subparsers.add_parser('dynamic', help='Run othertales Dynamic Pipeline for any URL')
    dynamic_parser.add_argument('url', nargs='?', help='URL to create datasets from')
    dynamic_parser.add_argument('--output-dir', help='Output directory for dynamic datasets')
    
    hmrc_parser = subparsers.add_parser('hmrc', help='Run HMRC tax documentation scraper')
    hmrc_parser.add_argument('--output-dir', help='Output directory for HMRC documentation')
    hmrc_parser.add_argument('--max-documents', type=int, help='Maximum number of documents to download')
    hmrc_parser.add_argument('--discover-only', action='store_true', help='Only discover URLs, do not download')
    
    housing_parser = subparsers.add_parser('housing', help='Run housing legislation and case law pipeline')
    housing_parser.add_argument('--output-dir', help='Output directory for housing data')
    housing_parser.add_argument('--max-documents', type=int, help='Maximum number of documents to download')
    
    bailii_parser = subparsers.add_parser('bailii', help='Run BAILII case law scraper')
    bailii_parser.add_argument('--output-dir', help='Output directory for case law')
    bailii_parser.add_argument('--max-documents', type=int, help='Maximum number of documents to download')
    
    complete_parser = subparsers.add_parser('complete', help='Run complete data collection pipeline')
    complete_parser.add_argument('--output-dir', help='Output directory for all data')
    complete_parser.add_argument('--max-documents', type=int, help='Maximum number of documents to download')
    
    # Enhanced pipelines for LLM training
    enhanced_parser = subparsers.add_parser('enhanced-complete', help='Run enhanced complete pipeline for Legal Llama training')
    enhanced_parser.add_argument('--output-dir', help='Output directory for enhanced data')
    enhanced_parser.add_argument('--max-documents', type=int, help='Maximum number of documents to download')
    
    legal_enhancer_parser = subparsers.add_parser('legal-enhancer', help='Enhance legal datasets with reasoning patterns')
    legal_enhancer_parser.add_argument('--input-dir', default='generated', help='Input directory containing legal data')
    legal_enhancer_parser.add_argument('--output-dir', help='Output directory for enhanced data')
    
    tax_scenarios_parser = subparsers.add_parser('tax-scenarios', help='Generate tax calculation and optimization scenarios')
    tax_scenarios_parser.add_argument('--input-dir', default='generated', help='Input directory containing tax data')
    tax_scenarios_parser.add_argument('--output-dir', help='Output directory for tax scenarios')
    
    advanced_qa_parser = subparsers.add_parser('advanced-qa', help='Generate advanced multi-step Q&A')
    advanced_qa_parser.add_argument('--input-dir', default='generated', help='Input directory containing legal and tax data')
    advanced_qa_parser.add_argument('--output-dir', help='Output directory for advanced Q&A')
    
    llama_optimizer_parser = subparsers.add_parser('llama-optimizer', help='Optimize datasets for Legal Llama 3.1 training')
    llama_optimizer_parser.add_argument('--input-dir', default='generated', help='Input directory containing all enhanced data')
    llama_optimizer_parser.add_argument('--output-dir', help='Output directory for Legal Llama training data')
    
    # Original utilities
    qa_parser = subparsers.add_parser('qa-generator', help='Generate Q&A pairs from collected data')
    qa_parser.add_argument('--input-dir', help='Input directory containing legal documents')
    qa_parser.add_argument('--output-dir', help='Output directory for Q&A pairs')
    
    db_parser = subparsers.add_parser('db-ingestion', help='Ingest data into databases')
    db_parser.add_argument('--input-dir', help='Input directory containing data to ingest')
    
    # Copyright pipeline
    copyright_parser = subparsers.add_parser('copyright', help='Run copyright law pipeline')
    copyright_parser.add_argument('--output-dir', help='Output directory for copyright data')
    copyright_parser.add_argument('--max-documents', type=int, help='Maximum number of documents to download')
    
    # Interactive menu
    menu_parser = subparsers.add_parser('menu', help='Show interactive menu')
    
    args = parser.parse_args()
    
    if not args.pipeline or args.pipeline == 'menu':
        show_interactive_menu()
        return
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Ensure generated directory exists
    os.makedirs('generated', exist_ok=True)
    
    # Route to appropriate pipeline
    if args.pipeline == 'dynamic':
        run_dynamic_pipeline(args)
    elif args.pipeline == 'hmrc':
        run_hmrc_scraper(args)
    elif args.pipeline == 'housing':
        run_housing_pipeline(args)
    elif args.pipeline == 'bailii':
        run_bailii_scraper(args)
    elif args.pipeline == 'complete':
        run_complete_pipeline(args)
    elif args.pipeline == 'enhanced-complete':
        run_enhanced_complete_pipeline(args)
    elif args.pipeline == 'legal-enhancer':
        run_legal_reasoning_enhancer(args)
    elif args.pipeline == 'tax-scenarios':
        run_tax_scenario_generator(args)
    elif args.pipeline == 'advanced-qa':
        run_advanced_qa_generator(args)
    elif args.pipeline == 'llama-optimizer':
        run_llama_training_optimizer(args)
    elif args.pipeline == 'qa-generator':
        run_qa_generator(args)
    elif args.pipeline == 'db-ingestion':
        run_database_ingestion(args)
    elif args.pipeline == 'copyright':
        run_copyright_pipeline(args)
    else:
        print(f"Unknown pipeline: {args.pipeline}")
        parser.print_help()

if __name__ == "__main__":
    main()