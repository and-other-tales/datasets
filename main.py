#!/usr/bin/env python3
"""
Legal Llama - Main Entry Point

This script provides a unified interface to run various legal data collection pipelines
and enhanced dataset generation for training domain-specialist Legal Llama models.
"""

import os
import sys
import argparse
import curses
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Callable

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.version import get_version_info, print_version_info
from utils.dataset_manager import DatasetManager

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
    # Check if we're already in a curses context
    import sys
    if hasattr(sys, '_curses_active') and sys._curses_active:
        # We're already in curses, use the standard main
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
    else:
        # Use the enhanced HMRC curses wrapper
        from utils.hmrc_curses_wrapper import run_hmrc_scraper_with_curses
        from pipelines.hmrc_scraper import HMRCScraper
        
        def hmrc_wrapper():
            output_dir = args.output_dir or 'generated/hmrc_documentation'
            scraper = HMRCScraper(output_dir)
            
            if args.discover_only:
                scraper.run_comprehensive_discovery()
            else:
                scraper.run_comprehensive_discovery()
                scraper.download_all_documents(args.max_documents)
                scraper.generate_summary()
                scraper.create_training_datasets()
        
        run_hmrc_scraper_with_curses(hmrc_wrapper)

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
    print("🔶 Pipeline Control: Press P to pause/resume, A to update databases (when paused), D to create dataset (when paused), Q to quit")
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

def run_dataset_manager(args):
    """Run interactive dataset management system"""
    from utils.dataset_manager import DatasetManager
    import curses
    import json
    
    def dataset_manager_menu(stdscr):
        """Dataset management menu interface"""
        manager = DatasetManager()
        menu = CursesMenu(stdscr)
        
        # Current dataset selection
        current_dataset = None
        
        def refresh_menu():
            """Refresh the menu with current dataset info"""
            menu.menu_items.clear()
            menu.current_row = 0
            
            # Dataset selection
            menu.add_category("DATASET SELECTION")
            
            # List available datasets
            datasets = manager.list_datasets()
            if datasets:
                for ds in datasets:
                    name = ds['name']
                    format_type = ds.get('format', 'unknown')
                    rows = ds.get('num_rows', 'unknown')
                    size = ds.get('size_bytes', 0)
                    size_mb = size / (1024 * 1024) if size else 0
                    
                    title = f"{name} ({format_type})"
                    desc = f"{rows} rows, {size_mb:.1f} MB"
                    
                    def select_dataset(ds_name=name):
                        nonlocal current_dataset
                        current_dataset = ds_name
                        curses.endwin()
                        print(f"Selected dataset: {ds_name}")
                        input("Press Enter to continue...")
                        stdscr.clear()
                        stdscr.refresh()
                        refresh_menu()
                    
                    menu.add_item(title, desc, select_dataset)
            else:
                menu.add_item("No datasets found", "Create datasets using pipelines first", lambda: None)
            
            menu.add_separator()
            
            # Dataset operations
            menu.add_category("DATASET OPERATIONS")
            
            if current_dataset:
                menu.add_item(f"Current: {current_dataset}", "Selected dataset", lambda: None)
                menu.add_separator()
                
                # View metadata
                def view_metadata():
                    curses.endwin()
                    try:
                        metadata = manager.get_dataset_metadata(current_dataset)
                        print(f"\n=== Metadata for {current_dataset} ===")
                        import json
                        print(json.dumps(metadata, indent=2))
                    except Exception as e:
                        print(f"Error: {e}")
                    input("\nPress Enter to continue...")
                    stdscr.clear()
                    stdscr.refresh()
                
                menu.add_item("View Metadata", "Display dataset metadata and statistics", view_metadata)
                
                # Add data
                def add_data():
                    curses.endwin()
                    print(f"\n=== Add Data to {current_dataset} ===")
                    print("1. Add from JSON file")
                    print("2. Add manual entry")
                    choice = input("Select option (1-2): ").strip()
                    
                    try:
                        if choice == "1":
                            json_path = input("Enter JSON file path: ").strip()
                            import json
                            with open(json_path, 'r') as f:
                                new_data = json.load(f)
                            manager.add_to_dataset(current_dataset, new_data)
                            print("Data added successfully!")
                        elif choice == "2":
                            print("Enter data as JSON (single object or array):")
                            json_str = input().strip()
                            new_data = json.loads(json_str)
                            manager.add_to_dataset(current_dataset, new_data)
                            print("Data added successfully!")
                    except Exception as e:
                        print(f"Error: {e}")
                    
                    input("\nPress Enter to continue...")
                    stdscr.clear()
                    stdscr.refresh()
                
                menu.add_item("Add Data", "Add new data to the dataset", add_data)
                
                # Edit fields
                def edit_fields():
                    curses.endwin()
                    print(f"\n=== Edit Fields in {current_dataset} ===")
                    try:
                        # Load dataset to show fields
                        dataset = manager.load_dataset(manager.datasets_dir / current_dataset)
                        print("Available fields:", dataset.column_names)
                        
                        print("\n1. Add new field")
                        print("2. Remove field")
                        print("3. Transform existing field")
                        choice = input("Select option (1-3): ").strip()
                        
                        if choice == "1":
                            field_name = input("Enter new field name: ").strip()
                            default_value = input("Enter default value for all rows: ").strip()
                            manager.add_dataset_field(current_dataset, field_name, 
                                                    lambda row: default_value)
                            print(f"Added field '{field_name}'")
                        elif choice == "2":
                            field_name = input("Enter field name to remove: ").strip()
                            manager.remove_dataset_field(current_dataset, field_name)
                            print(f"Removed field '{field_name}'")
                        elif choice == "3":
                            field_name = input("Enter field name to transform: ").strip()
                            print("Enter Python expression to transform (use 'x' for current value):")
                            transform_expr = input("Transform: ").strip()
                            transform_func = eval(f"lambda x: {transform_expr}")
                            manager.edit_dataset_field(current_dataset, field_name, transform_func)
                            print(f"Transformed field '{field_name}'")
                    except Exception as e:
                        print(f"Error: {e}")
                    
                    input("\nPress Enter to continue...")
                    stdscr.clear()
                    stdscr.refresh()
                    refresh_menu()
                
                menu.add_item("Edit Fields", "Add, remove, or transform dataset fields", edit_fields)
                
                # Update metadata
                def update_metadata():
                    curses.endwin()
                    print(f"\n=== Update Metadata for {current_dataset} ===")
                    print("Enter metadata as JSON object:")
                    try:
                        json_str = input().strip()
                        metadata = json.loads(json_str)
                        manager.update_dataset_metadata(current_dataset, metadata)
                        print("Metadata updated successfully!")
                    except Exception as e:
                        print(f"Error: {e}")
                    
                    input("\nPress Enter to continue...")
                    stdscr.clear()
                    stdscr.refresh()
                
                menu.add_item("Update Metadata", "Update custom metadata for the dataset", update_metadata)
                
                # Export dataset
                def export_dataset():
                    curses.endwin()
                    print(f"\n=== Export {current_dataset} ===")
                    print("1. Export as Parquet")
                    print("2. Export as JSON")
                    print("3. Export as CSV")
                    choice = input("Select format (1-3): ").strip()
                    
                    try:
                        format_map = {"1": "parquet", "2": "json", "3": "csv"}
                        if choice in format_map:
                            export_format = format_map[choice]
                            output_path = manager.export_dataset(current_dataset, export_format)
                            print(f"Exported to: {output_path}")
                    except Exception as e:
                        print(f"Error: {e}")
                    
                    input("\nPress Enter to continue...")
                    stdscr.clear()
                    stdscr.refresh()
                
                menu.add_item("Export Dataset", "Export dataset to different formats", export_dataset)
                
                # Delete dataset
                def delete_dataset():
                    nonlocal current_dataset
                    curses.endwin()
                    try:
                        if manager.delete_dataset(current_dataset, confirm=True):
                            print(f"Dataset '{current_dataset}' deleted")
                            current_dataset = None
                    except Exception as e:
                        print(f"Error: {e}")
                    
                    input("\nPress Enter to continue...")
                    stdscr.clear()
                    stdscr.refresh()
                    refresh_menu()
                
                menu.add_item("Delete Dataset", "Delete the current dataset", delete_dataset)
                
            else:
                menu.add_item("Select a dataset first", "Choose a dataset from the list above", lambda: None)
            
            menu.add_separator()
            
            # Cache operations
            menu.add_category("CACHE MANAGEMENT")
            
            def clear_all_cache():
                curses.endwin()
                response = input("Clear all dataset cache? (y/N): ").strip().lower()
                if response == 'y':
                    try:
                        manager.clear_cache()
                        print("Cache cleared successfully!")
                    except Exception as e:
                        print(f"Error: {e}")
                
                input("\nPress Enter to continue...")
                stdscr.clear()
                stdscr.refresh()
            
            menu.add_item("Clear All Cache", "Remove all cached dataset files", clear_all_cache)
            
            if current_dataset:
                def clear_dataset_cache():
                    curses.endwin()
                    try:
                        manager.clear_cache(current_dataset)
                        print(f"Cache cleared for {current_dataset}")
                    except Exception as e:
                        print(f"Error: {e}")
                    
                    input("\nPress Enter to continue...")
                    stdscr.clear()
                    stdscr.refresh()
                
                menu.add_item(f"Clear Cache for {current_dataset}", 
                            "Remove cached files for current dataset", clear_dataset_cache)
            
            menu.add_separator()
            
            # Create new dataset
            menu.add_category("CREATE DATASET")
            
            def create_dataset():
                curses.endwin()
                print("\n=== Create New Dataset ===")
                name = input("Enter dataset name: ").strip()
                if not name:
                    print("Dataset name required")
                else:
                    print("Enter initial data as JSON array:")
                    try:
                        json_str = input().strip()
                        data = json.loads(json_str)
                        from datasets import Dataset
                        dataset = Dataset.from_list(data if isinstance(data, list) else [data])
                        manager.save_dataset(dataset, name)
                        print(f"Created dataset '{name}'")
                        nonlocal current_dataset
                        current_dataset = name
                    except Exception as e:
                        print(f"Error: {e}")
                
                input("\nPress Enter to continue...")
                stdscr.clear()
                stdscr.refresh()
                refresh_menu()
            
            menu.add_item("Create New Dataset", "Create a new empty dataset", create_dataset)
            
            menu.add_separator()
            menu.add_item("Back to Main Menu", "Return to main menu", lambda: "exit")
        
        # Initial menu setup
        refresh_menu()
        
        while True:
            action = menu.run()
            if action == "exit" or action is None:
                break
            elif callable(action):
                result = action()
                if result == "exit":
                    break
    
    # Run the dataset manager menu
    try:
        curses.wrapper(dataset_manager_menu)
    except Exception as e:
        print(f"Dataset manager error: {e}")
        input("Press Enter to continue...")

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

def run_enhanced_legal_pipeline(args):
    """Run the new enhanced legal pipeline with GUIDANCE.md implementation"""
    from pipelines.enhanced_legal_pipeline import main as enhanced_main
    
    # Set up arguments for enhanced legal pipeline
    enhanced_args = []
    
    if args.input_dir:
        enhanced_args.extend(['--input-dir', args.input_dir])
    else:
        enhanced_args.extend(['--input-dir', args.output_dir or 'generated'])
    
    if args.output_dir:
        enhanced_args.extend(['--output-dir', args.output_dir])
    
    if args.max_documents:
        enhanced_args.extend(['--max-documents', str(args.max_documents)])
    
    # Override sys.argv for the enhanced pipeline
    original_argv = sys.argv
    sys.argv = ['enhanced_legal_pipeline.py'] + enhanced_args
    
    try:
        enhanced_main()
    except ImportError:
        print("Enhanced legal pipeline main function not found. Running enhanced pipeline directly...")
        import pipelines.enhanced_legal_pipeline
    finally:
        sys.argv = original_argv

def show_interactive_menu():
    """Show interactive curses-based menu for pipeline selection"""
    def create_menu(stdscr):
        menu = CursesMenu(stdscr)
        
        # Data Collection Pipelines
        menu.add_category("DATA COLLECTION PIPELINES")
        menu.add_item("Dynamic Pipeline (Any URL)", 
                     "Generate datasets from any URL using AI analysis", 
                     lambda: _run_with_menu_args(run_dynamic_pipeline, "Dynamic Pipeline"))
        menu.add_item("HMRC Tax Documentation Scraper", 
                     "Collect HMRC tax documentation from gov.uk", 
                     lambda: _run_with_menu_args(run_hmrc_scraper, "HMRC Scraper"))
        menu.add_item("Housing Legislation & Case Law", 
                     "Collect housing legislation and case law", 
                     lambda: _run_with_menu_args(run_housing_pipeline, "Housing Pipeline"))
        menu.add_item("BAILII Case Law Scraper", 
                     "Scrape case law from BAILII database", 
                     lambda: _run_with_menu_args(run_bailii_scraper, "BAILII Scraper"))
        menu.add_item("Copyright Law Pipeline", 
                     "Collect copyright and IP law documentation", 
                     lambda: _run_with_menu_args(run_copyright_pipeline, "Copyright Pipeline"))
        menu.add_item("Complete Data Collection Pipeline", 
                     "Run all data collection pipelines with pause controls", 
                     lambda: _run_with_menu_args(run_complete_pipeline, "Complete Pipeline"))
        
        menu.add_separator()
        
        # Dataset Enhancement
        menu.add_category("DATASET ENHANCEMENT (for LLM Training)")
        menu.add_item("Legal Reasoning Enhancer", 
                     "Enhance legal datasets with reasoning patterns", 
                     lambda: _run_with_menu_args(run_legal_reasoning_enhancer, "Legal Reasoning Enhancer"))
        menu.add_item("Tax Scenario Generator", 
                     "Generate tax calculation and optimization scenarios", 
                     lambda: _run_with_menu_args(run_tax_scenario_generator, "Tax Scenario Generator"))
        menu.add_item("Advanced Q&A Generator", 
                     "Generate advanced multi-step Q&A pairs", 
                     lambda: _run_with_menu_args(run_advanced_qa_generator, "Advanced Q&A Generator"))
        menu.add_item("Legal Llama Training Optimizer", 
                     "Optimize datasets for Legal Llama 3.1 training", 
                     lambda: _run_with_menu_args(run_llama_training_optimizer, "Legal Llama Training Optimizer"))
        
        menu.add_separator()
        
        # Complete Workflows
        menu.add_category("COMPLETE WORKFLOWS")
        menu.add_item("Enhanced Complete Pipeline (All Steps)", 
                     "Run full data collection and enhancement pipeline", 
                     lambda: _run_with_menu_args(run_enhanced_complete_pipeline, "Enhanced Complete Pipeline"))
        menu.add_item("Production Legal AI Pipeline", 
                     "Run production legal AI pipeline with GUIDANCE.md implementation", 
                     lambda: _run_with_menu_args(run_enhanced_legal_pipeline, "Production Legal AI Pipeline"))
        menu.add_item("Q&A Generation Only", 
                     "Generate Q&A pairs from existing data", 
                     lambda: _run_with_menu_args(run_qa_generator, "Q&A Generator"))
        menu.add_item("Database Ingestion", 
                     "Ingest data into MongoDB, Neo4j, and Pinecone", 
                     lambda: _run_with_menu_args(run_database_ingestion, "Database Ingestion"))
        
        menu.add_separator()
        
        # Dataset Management
        menu.add_category("DATASET MANAGEMENT")
        menu.add_item("Dataset Manager", 
                     "Manage existing datasets - add, delete, edit, export", 
                     lambda: _run_with_menu_args(run_dataset_manager, "Dataset Manager"))
        
        menu.add_separator()
        
        # Other Options
        menu.add_category("OTHER OPTIONS")
        menu.add_item("Show Pipeline Status", 
                     "View status of data directories and files", 
                     lambda: _show_pipeline_status())
        menu.add_item("View Documentation", 
                     "Show quick documentation and help", 
                     lambda: _show_documentation())
        menu.add_item("Manage Credentials", 
                     "Edit database and API credentials", 
                     lambda: _manage_credentials())
        
        menu.add_separator()
        menu.add_item("Exit", "Exit the application", lambda: None)
        
        while True:
            action = menu.run()
            if action is None:
                break
            
            # Execute the selected action
            try:
                curses.endwin()  # Temporarily exit curses mode
                result = action()
                if result is None and action.__name__ == '<lambda>':
                    # If it's the exit option
                    break
                input("\nPress Enter to return to menu...")
            except Exception as e:
                print(f"Error: {e}")
                input("Press Enter to continue...")
            finally:
                # Restart curses
                stdscr.clear()
                stdscr.refresh()
        
        # Final cleanup
        curses.endwin()
    
    try:
        curses.wrapper(create_menu)
    except Exception as e:
        print(f"Menu error: {e}")
        print("Falling back to text menu...")
        _show_text_menu_fallback()

def _show_text_menu_fallback():
    """Fallback text menu if curses fails"""
    while True:
        print("\n" + "="*60)
        print("    ░█▀▄▒▄▀▄░▀█▀▒▄▀▄░▄▀▀▒██▀░▀█▀░▄▀▀")
        print("    ▒█▄▀░█▀█░▒█▒░█▀█▒▄██░█▄▄░▒█▒▒▄██")
        print("="*60)
        print()
        print("1. Dynamic Pipeline (Any URL)")
        print("2. HMRC Tax Documentation Scraper")
        print("3. Housing Legislation & Case Law Pipeline")
        print("4. BAILII Case Law Scraper")
        print("5. Copyright Law Pipeline")
        print("6. Complete Data Collection Pipeline")
        print("7. Legal Reasoning Enhancer")
        print("8. Tax Scenario Generator")
        print("9. Advanced Q&A Generator")
        print("10. Legal Llama Training Optimizer")
        print("11. Enhanced Complete Pipeline")
        print("12. Production Legal AI Pipeline")
        print("13. Q&A Generation Only")
        print("14. Database Ingestion")
        print("15. Dataset Manager")
        print("16. Show Pipeline Status")
        print("17. View Documentation")
        print("18. Manage Credentials")
        print("0. Exit")
        print()
        
        try:
            choice = input("Select an option (0-18): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                _run_with_menu_args(run_dynamic_pipeline, "Dynamic Pipeline")
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
                _run_with_menu_args(run_llama_training_optimizer, "Legal Llama Training Optimizer")
            elif choice == "11":
                _run_with_menu_args(run_enhanced_complete_pipeline, "Enhanced Complete Pipeline")
            elif choice == "12":
                _run_with_menu_args(run_enhanced_legal_pipeline, "Production Legal AI Pipeline")
            elif choice == "13":
                _run_with_menu_args(run_qa_generator, "Q&A Generator")
            elif choice == "14":
                _run_with_menu_args(run_database_ingestion, "Database Ingestion")
            elif choice == "15":
                _run_with_menu_args(run_dataset_manager, "Dataset Manager")
            elif choice == "16":
                _show_pipeline_status()
            elif choice == "17":
                _show_documentation()
            elif choice == "18":
                _manage_credentials()
            else:
                print("Invalid choice. Please select a number between 0-18.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def _run_with_menu_args(pipeline_func, pipeline_name):
    """Run a pipeline with interactive argument collection in curses"""
    # Special handling for HMRC scraper
    if pipeline_name == "HMRC Tax Documentation Scraper":
        # Set flag to indicate we're in curses
        sys._curses_active = True
        from utils.curses_pipeline_runner import run_pipeline_in_curses
    else:
        from utils.curses_pipeline_runner import run_pipeline_in_curses
    
    def args_collector(input_win):
        """Collect arguments within curses interface"""
        curses.curs_set(1)  # Show cursor for input
        
        class Args:
            def __init__(self):
                self.input_dir = None
                self.output_dir = None
                self.max_documents = None
                self.discover_only = False
                self.url = None
        
        args = Args()
        y_pos = 2
        
        # Special handling for Dynamic Pipeline
        if pipeline_name == "Dynamic Pipeline":
            try:
                max_y, max_x = input_win.getmaxyx()
                if y_pos < max_y - 2 and max_x > 40:
                    input_win.addstr(y_pos, 2, "Enter URL to create datasets from:"[:max_x-4])
                    y_pos += 1
                    input_win.addstr(y_pos, 2, "URL: ")
                    input_win.refresh()
            except curses.error:
                # Fallback without colors if there's an issue
                max_y, max_x = input_win.getmaxyx()
                if y_pos < max_y - 2 and max_x > 10:
                    input_win.addstr(y_pos, 2, "Enter URL:"[:max_x-4])
                    y_pos += 1
                    input_win.addstr(y_pos, 2, "URL: ")
                    input_win.refresh()
            
            # Get URL input with bounds checking
            try:
                curses.echo()
                max_y, max_x = input_win.getmaxyx()
                if y_pos < max_y - 1 and max_x > 10:
                    url = input_win.getstr(y_pos, 7, min(50, max_x - 10)).decode('utf-8').strip()
                else:
                    url = ""
                curses.noecho()
                
                if not url:
                    raise ValueError("URL is required for dynamic pipeline")
                args.url = url
                y_pos += 2
            except (curses.error, ValueError) as e:
                curses.noecho()
                raise e
            
            # Output directory input with bounds checking
            try:
                max_y, max_x = input_win.getmaxyx()
                if y_pos < max_y - 2 and max_x > 20:
                    input_win.addstr(y_pos, 2, "Output dir (Enter=default):"[:max_x-4])
                    y_pos += 1
                    input_win.addstr(y_pos, 2, "Dir: ")
                    input_win.refresh()
                    
                    curses.echo()
                    output_dir = input_win.getstr(y_pos, 7, min(30, max_x - 10)).decode('utf-8').strip()
                    curses.noecho()
                else:
                    output_dir = ""
            except curses.error:
                curses.noecho()
                output_dir = ""
            
            if output_dir:
                args.output_dir = output_dir
        
        elif pipeline_name == "Dataset Manager":
            # No special arguments needed for dataset manager
            pass
        else:
            # Get common arguments with bounds checking
            if pipeline_name in ["HMRC Scraper", "Housing Pipeline", "BAILII Scraper", "Complete Pipeline", "Copyright Pipeline"]:
                try:
                    max_y, max_x = input_win.getmaxyx()
                    if y_pos < max_y - 2 and max_x > 25:
                        input_win.addstr(y_pos, 2, "Max documents (Enter=all):"[:max_x-4])
                        y_pos += 1
                        input_win.addstr(y_pos, 2, "Max: ")
                        input_win.refresh()
                        
                        curses.echo()
                        max_docs = input_win.getstr(y_pos, 7, min(10, max_x - 10)).decode('utf-8').strip()
                        curses.noecho()
                        
                        if max_docs:
                            try:
                                args.max_documents = int(max_docs)
                            except ValueError:
                                pass  # Use default
                        y_pos += 2
                except curses.error:
                    pass
                
                if pipeline_name == "HMRC Scraper":
                    try:
                        max_y, max_x = input_win.getmaxyx()
                        if y_pos < max_y - 2 and max_x > 20:
                            input_win.addstr(y_pos, 2, "Discovery only? (y/N):"[:max_x-4])
                            y_pos += 1
                            input_win.addstr(y_pos, 2, "Discover: ")
                            input_win.refresh()
                            
                            curses.echo()
                            discover = input_win.getstr(y_pos, 11, 1).decode('utf-8').strip().lower()
                            curses.noecho()
                            
                            args.discover_only = discover == 'y'
                            y_pos += 2
                    except curses.error:
                        pass
            
            # Output directory with bounds checking
            try:
                max_y, max_x = input_win.getmaxyx()
                if y_pos < max_y - 2 and max_x > 20:
                    input_win.addstr(y_pos, 2, "Output dir (Enter=default):"[:max_x-4])
                    y_pos += 1
                    input_win.addstr(y_pos, 2, "Output: ")
                    input_win.refresh()
                    
                    curses.echo()
                    output_dir = input_win.getstr(y_pos, 9, min(30, max_x - 12)).decode('utf-8').strip()
                    curses.noecho()
                    
                    if output_dir:
                        args.output_dir = output_dir
                    y_pos += 2
            except curses.error:
                pass
            
            # Input directory with bounds checking
            try:
                max_y, max_x = input_win.getmaxyx()
                if y_pos < max_y - 2 and max_x > 20:
                    input_win.addstr(y_pos, 2, "Input dir (Enter=default):"[:max_x-4])
                    y_pos += 1
                    input_win.addstr(y_pos, 2, "Input: ")
                    input_win.refresh()
                    
                    curses.echo()
                    input_dir = input_win.getstr(y_pos, 8, min(30, max_x - 11)).decode('utf-8').strip()
                    curses.noecho()
                    
                    if input_dir:
                        args.input_dir = input_dir
            except curses.error:
                pass
        
        curses.curs_set(0)  # Hide cursor
        return args
    
    # Run the pipeline in curses
    run_pipeline_in_curses(pipeline_name, pipeline_func, args_collector)

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

def _load_env_credentials() -> Dict[str, str]:
    """Load credentials from .env file"""
    env_file = Path('.env')
    credentials = {}
    
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        credentials[key.strip()] = value.strip()
        except Exception as e:
            print(f"Error reading .env file: {e}")
    
    return credentials

def _save_env_credentials(credentials: Dict[str, str]):
    """Save credentials to .env file"""
    env_file = Path('.env')
    
    try:
        # Read existing file to preserve comments and structure
        existing_lines = []
        if env_file.exists():
            with open(env_file, 'r') as f:
                existing_lines = f.readlines()
        
        # Write updated credentials
        with open(env_file, 'w') as f:
            for line in existing_lines:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key = line.split('=', 1)[0].strip()
                    if key in credentials:
                        f.write(f"{key}={credentials[key]}\n")
                    else:
                        f.write(line + '\n')
                else:
                    f.write(line + '\n')
            
            # Add any new credentials that weren't in the original file
            existing_keys = set()
            for line in existing_lines:
                if '=' in line and not line.strip().startswith('#'):
                    existing_keys.add(line.split('=', 1)[0].strip())
            
            for key, value in credentials.items():
                if key not in existing_keys:
                    f.write(f"{key}={value}\n")
        
        print("✅ Credentials saved successfully!")
        
    except Exception as e:
        print(f"❌ Error saving credentials: {e}")

class MenuItem:
    """Represents a menu item with title, description, and action"""
    def __init__(self, title: str, description: str, action: Callable, category: str = ""):
        self.title = title
        self.description = description
        self.action = action
        self.category = category

class CursesMenu:
    """Modern curses-based menu interface"""
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.current_row = 0
        self.top_row = 0
        self.menu_items = []
        self.categories = []
        self.max_visible_items = 0
        
        # Initialize colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Selected item
        curses.init_pair(2, curses.COLOR_CYAN, -1)    # Category headers
        curses.init_pair(3, curses.COLOR_GREEN, -1)   # Active items
        curses.init_pair(4, curses.COLOR_YELLOW, -1)  # Descriptions
        curses.init_pair(5, curses.COLOR_RED, -1)     # Exit items
        
        # Hide cursor
        curses.curs_set(0)
        
    def add_category(self, category_name: str):
        """Add a category separator"""
        self.categories.append(len(self.menu_items))
        self.menu_items.append(MenuItem(f"=== {category_name} ===", "", None, "category"))
        
    def add_item(self, title: str, description: str, action: Callable, category: str = ""):
        """Add a menu item"""
        self.menu_items.append(MenuItem(title, description, action, category))
        
    def add_separator(self):
        """Add a visual separator"""
        self.menu_items.append(MenuItem("", "", None, "separator"))
        
    def draw_header(self):
        """Draw the header"""
        height, width = self.stdscr.getmaxyx()
        
        # Clear header area
        for i in range(6):
            self.stdscr.move(i, 0)
            self.stdscr.clrtoeol()
        
        # ASCII Art Title
        ascii_title_line1 = "░█▀▄▒▄▀▄░▀█▀▒▄▀▄░▄▀▀▒██▀░▀█▀░▄▀▀"
        ascii_title_line2 = "▒█▄▀░█▀█░▒█▒░█▀█▒▄██░█▄▄░▒█▒▒▄██"
        
        # Center the ASCII art
        self.stdscr.addstr(1, (width - len(ascii_title_line1)) // 2, ascii_title_line1, curses.A_BOLD)
        self.stdscr.addstr(2, (width - len(ascii_title_line2)) // 2, ascii_title_line2, curses.A_BOLD)
        
        # Subtitle
        subtitle = "Other Tales Datasets Generation Tool"
        self.stdscr.addstr(3, (width - len(subtitle)) // 2, subtitle, curses.color_pair(4))
        
        # Controls
        controls = "↑↓: Navigate  ENTER: Select  Q: Quit"
        self.stdscr.addstr(5, (width - len(controls)) // 2, controls, curses.color_pair(4))
        
    def draw_menu(self):
        """Draw the menu items"""
        height, width = self.stdscr.getmaxyx()
        self.max_visible_items = height - 8  # Reserve space for header and footer
        
        menu_start_y = 7
        
        # Clear menu area
        for i in range(menu_start_y, height - 2):
            self.stdscr.move(i, 0)
            self.stdscr.clrtoeol()
        
        # Determine visible range
        if self.current_row >= self.top_row + self.max_visible_items:
            self.top_row = self.current_row - self.max_visible_items + 1
        elif self.current_row < self.top_row:
            self.top_row = self.current_row
            
        # Draw visible menu items
        for i, item_idx in enumerate(range(self.top_row, min(self.top_row + self.max_visible_items, len(self.menu_items)))):
            y = menu_start_y + i
            item = self.menu_items[item_idx]
            
            if item.category == "category":
                # Category header
                self.stdscr.addstr(y, 2, item.title, curses.color_pair(2) | curses.A_BOLD)
            elif item.category == "separator":
                # Separator line
                self.stdscr.addstr(y, 2, "─" * (width - 4))
            else:
                # Regular menu item
                is_selected = item_idx == self.current_row
                
                if is_selected:
                    # Highlight selected item
                    self.stdscr.addstr(y, 0, " " * width, curses.color_pair(1))
                    self.stdscr.addstr(y, 2, f"▶ {item.title}", curses.color_pair(1) | curses.A_BOLD)
                    
                    # Show description at bottom
                    if item.description:
                        desc_y = height - 2
                        self.stdscr.move(desc_y, 0)
                        self.stdscr.clrtoeol()
                        desc_text = f"Info: {item.description}"
                        if len(desc_text) > width - 2:
                            desc_text = desc_text[:width - 5] + "..."
                        self.stdscr.addstr(desc_y, 2, desc_text, curses.color_pair(4))
                else:
                    # Regular item
                    color = curses.color_pair(3)
                    if "exit" in item.title.lower() or "quit" in item.title.lower():
                        color = curses.color_pair(5)
                    
                    self.stdscr.addstr(y, 4, item.title, color)
        
        # Draw scrollbar if needed
        if len(self.menu_items) > self.max_visible_items:
            self.draw_scrollbar(menu_start_y, self.max_visible_items)
            
    def draw_scrollbar(self, start_y: int, visible_items: int):
        """Draw a scrollbar on the right side"""
        height, width = self.stdscr.getmaxyx()
        scrollbar_x = width - 2
        
        # Calculate scrollbar position
        total_items = len(self.menu_items)
        thumb_size = max(1, (visible_items * visible_items) // total_items)
        thumb_pos = (self.top_row * visible_items) // total_items
        
        # Draw scrollbar track
        for i in range(visible_items):
            self.stdscr.addstr(start_y + i, scrollbar_x, "│", curses.color_pair(4))
        
        # Draw scrollbar thumb
        for i in range(thumb_size):
            if thumb_pos + i < visible_items:
                self.stdscr.addstr(start_y + thumb_pos + i, scrollbar_x, "█", curses.color_pair(2))
    
    def find_next_selectable(self, start_idx: int, direction: int) -> int:
        """Find the next selectable menu item"""
        items_count = len(self.menu_items)
        if items_count == 0:
            return 0
            
        # Handle initial case where start_idx is -1
        if start_idx == -1:
            start_idx = 0 if direction > 0 else items_count - 1
        
        idx = start_idx
        visited = set()
        
        while idx not in visited:
            visited.add(idx)
            idx = (idx + direction) % items_count
            
            if 0 <= idx < items_count:
                item = self.menu_items[idx]
                if item.action is not None and item.category not in ["category", "separator"]:
                    return idx
        
        # If no selectable items found, return the first valid index
        for i, item in enumerate(self.menu_items):
            if item.action is not None and item.category not in ["category", "separator"]:
                return i
                
        return 0
    
    def run(self) -> Optional[Callable]:
        """Run the menu and return selected action"""
        # Find first selectable item
        self.current_row = self.find_next_selectable(-1, 1)
        
        while True:
            self.stdscr.clear()
            self.draw_header()
            self.draw_menu()
            self.stdscr.refresh()
            
            try:
                key = self.stdscr.getch()
                
                if key == curses.KEY_UP:
                    self.current_row = self.find_next_selectable(self.current_row, -1)
                elif key == curses.KEY_DOWN:
                    self.current_row = self.find_next_selectable(self.current_row, 1)
                elif key == ord('\n') or key == curses.KEY_ENTER:
                    # Return selected action
                    if self.current_row < len(self.menu_items):
                        item = self.menu_items[self.current_row]
                        if item.action:
                            return item.action
                elif key == ord('q') or key == ord('Q'):
                    return None
                    
            except KeyboardInterrupt:
                return None

def _manage_credentials():
    """Interactive credential management"""
    def manage_credentials_curses(stdscr):
        menu = CursesMenu(stdscr)
        
        # Load current credentials
        credentials = _load_env_credentials()
        
        # Define expected credentials with descriptions
        credential_definitions = {
            'MONGODB_CONNECTION_STRING': 'MongoDB Atlas connection string (mongodb+srv://...)',
            'MONGODB_DATABASE': 'MongoDB database name (default: legal_datasets)',
            'NEO4J_URI': 'Neo4j connection URI (bolt://...)',
            'NEO4J_USERNAME': 'Neo4j username (default: neo4j)',
            'NEO4J_PASSWORD': 'Neo4j password',
            'PINECONE_API_KEY': 'Pinecone API key',
            'PINECONE_ENVIRONMENT': 'Pinecone environment (default: us-west1-gcp)',
            'ANTHROPIC_API_KEY': 'Anthropic API key for Claude integration'
        }
        
        menu.add_category("DATABASE CREDENTIALS")
        
        def create_edit_action(key):
            def edit_credential():
                # Switch back to normal mode to get input
                curses.endwin()
                description = credential_definitions[key]
                current_value = credentials.get(key, '')
                print(f"\nEditing: {key}")
                print(f"Description: {description}")
                print(f"Current value: {'***Hidden***' if ('PASSWORD' in key or 'KEY' in key) and current_value else current_value}")
                
                new_value = input("Enter new value (press Enter to keep current): ").strip()
                if new_value:
                    credentials[key] = new_value
                    print(f"✅ Updated {key}")
                else:
                    print("Value unchanged")
                
                input("\nPress Enter to continue...")
                # Restart curses
                stdscr.clear()
                stdscr.refresh()
            return edit_credential
        
        # Add credential items
        for key, description in credential_definitions.items():
            current_value = credentials.get(key, 'Not set')
            if 'PASSWORD' in key or 'KEY' in key:
                display_value = '***Hidden***' if current_value != 'Not set' else 'Not set'
            else:
                display_value = current_value[:50] + '...' if len(current_value) > 50 else current_value
            
            title = f"{key}: {display_value}"
            menu.add_item(title, description, create_edit_action(key))
        
        menu.add_separator()
        
        def save_and_exit():
            curses.endwin()
            _save_env_credentials(credentials)
            print("Credentials saved. Restart applications to use new credentials.")
            input("Press Enter to continue...")
            stdscr.clear()
            stdscr.refresh()
            return "exit"
        
        def exit_without_saving():
            return "exit"
        
        menu.add_item("Save and Exit", "Save credentials to .env file and return to main menu", save_and_exit)
        menu.add_item("Exit without Saving", "Return to main menu without saving changes", exit_without_saving)
        
        while True:
            action = menu.run()
            if action is None or (callable(action) and action() == "exit"):
                break
    
    # Run the curses interface
    curses.wrapper(manage_credentials_curses)

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
  dataset-manager         - Manage existing datasets (add, delete, edit, export)
  menu                    - Show interactive menu

Examples:
  python main.py                                    # Show interactive menu
  python main.py menu                               # Show interactive menu
  python main.py enhanced-complete                  # Run full enhanced pipeline
  python main.py hmrc --max-documents 100          # Collect HMRC data
  python main.py llama-optimizer                   # Prepare for Legal Llama training
        """
    )
    
    # Add version argument
    parser.add_argument('--version', action='version', version=f'Legal Llama Datasets {get_version_info()["version"]}')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='pipeline', help='Pipeline to run')
    
    # Version subcommand for detailed info
    version_parser = subparsers.add_parser('version', help='Show detailed version information')
    version_parser.add_argument('--json', action='store_true', help='Output version info as JSON')
    
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
    
    # Production Legal AI Pipeline
    legal_ai_parser = subparsers.add_parser('legal-ai', help='Run production legal AI pipeline (GUIDANCE.md implementation)')
    legal_ai_parser.add_argument('--input-dir', default='generated', help='Input directory containing collected data')
    legal_ai_parser.add_argument('--output-dir', help='Output directory for legal AI system')
    legal_ai_parser.add_argument('--max-documents', type=int, help='Maximum number of documents to process')
    
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
    
    # Dataset manager
    dataset_manager_parser = subparsers.add_parser('dataset-manager', help='Manage existing datasets')
    
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
    if args.pipeline == 'version':
        if args.json:
            import json
            print(json.dumps(get_version_info(), indent=2))
        else:
            print_version_info()
        return
    elif args.pipeline == 'dynamic':
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
    elif args.pipeline == 'legal-ai':
        run_enhanced_legal_pipeline(args)
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
    elif args.pipeline == 'dataset-manager':
        run_dataset_manager(args)
    else:
        print(f"Unknown pipeline: {args.pipeline}")
        parser.print_help()

if __name__ == "__main__":
    main()