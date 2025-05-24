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
from typing import Dict, Optional, List, Tuple, Callable, Any
import json  # Fix: Import missing json module

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.version import get_version_info, print_version_info
from utils.dataset_manager import DatasetManager

# Add CursesMenu class definition
class CursesMenu:
    """Simple curses-based menu implementation"""
    
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.menu_items = []
        self.current_row = 0
        
        # Initialize colors if available
        if curses.has_colors():
            curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Selected item
            curses.init_pair(2, curses.COLOR_CYAN, -1)                   # Category headers
            curses.init_pair(3, curses.COLOR_GREEN, -1)                  # Active items
            curses.init_pair(4, curses.COLOR_YELLOW, -1)                 # Descriptions
            curses.init_pair(5, curses.COLOR_RED, -1)                    # Exit items
    
    def add_item(self, title, description, action):
        """Add a menu item"""
        self.menu_items.append({
            'type': 'item',
            'title': title,
            'description': description,
            'action': action
        })
    
    def add_category(self, title):
        """Add a category header"""
        self.menu_items.append({
            'type': 'category',
            'title': title
        })
    
    def add_separator(self):
        """Add a separator line"""
        self.menu_items.append({
            'type': 'separator'
        })
    
    def run(self):
        """Run the menu loop"""
        # Hide cursor
        curses.curs_set(0)
        
        # Enable key input
        self.stdscr.keypad(True)
        
        while True:
            # Clear screen
            self.stdscr.clear()
            
            # Get terminal dimensions
            height, width = self.stdscr.getmaxyx()
            
            # Display menu items
            max_display_items = height - 3  # Leave room for status line
            start_row = max(0, self.current_row - max_display_items // 2)
            
            # Ensure current_row is valid
            if self.current_row >= len(self.menu_items):
                self.current_row = len(self.menu_items) - 1
            if self.current_row < 0:
                self.current_row = 0
            
            # Display items
            for i, item in enumerate(self.menu_items[start_row:start_row + max_display_items]):
                y = i + 1  # Start at row 1
                
                if item['type'] == 'category':
                    # Display category header
                    attr = curses.color_pair(2) | curses.A_BOLD
                    self.stdscr.addstr(y, 1, item['title'].upper(), attr)
                
                elif item['type'] == 'separator':
                    # Display separator line
                    self.stdscr.addstr(y, 1, "-" * (width - 2))
                
                elif item['type'] == 'item':
                    # Display menu item
                    title = item['title']
                    
                    # Highlight current row
                    if start_row + i == self.current_row:
                        attr = curses.color_pair(1)
                        self.stdscr.addstr(y, 1, title[:width-2], attr)
                        
                        # Display description at bottom
                        desc = item['description']
                        if desc:
                            desc_y = height - 1
                            self.stdscr.addstr(desc_y, 1, desc[:width-2], curses.color_pair(4))
                    else:
                        # Normal item
                        if title.lower() == "exit":
                            attr = curses.color_pair(5)
                        else:
                            attr = curses.A_NORMAL
                        self.stdscr.addstr(y, 1, title[:width-2], attr)
            
            # Refresh screen
            self.stdscr.refresh()
            
            # Handle key presses
            key = self.stdscr.getch()
            
            if key == curses.KEY_UP:
                # Move up
                self.current_row = max(0, self.current_row - 1)
                # Skip non-selectable items
                while (self.current_row > 0 and 
                       self.menu_items[self.current_row]['type'] != 'item'):
                    self.current_row -= 1
            
            elif key == curses.KEY_DOWN:
                # Move down
                self.current_row = min(len(self.menu_items) - 1, self.current_row + 1)
                # Skip non-selectable items
                while (self.current_row < len(self.menu_items) - 1 and 
                       self.menu_items[self.current_row]['type'] != 'item'):
                    self.current_row += 1
            
            elif key == curses.KEY_ENTER or key in [10, 13]:
                # Execute action if it's a valid item
                if (0 <= self.current_row < len(self.menu_items) and 
                    self.menu_items[self.current_row]['type'] == 'item'):
                    action = self.menu_items[self.current_row]['action']
                    if action:
                        return action
            
            elif key == ord('q') or key == ord('Q'):
                # Quit
                return None

# Function to check if curses is active - fixes the sys._curses_active issue
def is_curses_active():
    """Check if curses is active without using sys._curses_active"""
    try:
        # If curses is active, this should throw an exception
        # if we're already in a curses environment
        import curses
        curses.initscr()
        curses.endwin()
        return False
    except Exception:
        # An exception likely means curses is already active
        return True

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
    from pipelines.hmrc_scraper import HMRCScraper
    
    # Determine if running in quick mode or full mode
    max_batches = None  # None means process ALL batches
    if hasattr(args, 'quick') and args.quick:
        max_batches = 20  # Quick mode: only process 20 batches (~2000 documents)
        print("Running in QUICK mode (limited discovery)")
    else:
        print("Running in FULL mode (comprehensive discovery - this may take several hours)")
    
    # Check if we should use curses
    use_curses = not (hasattr(args, 'no_curses') and args.no_curses)
    
    # Check if we're already in a curses context
    if is_curses_active():
        use_curses = False  # Don't use nested curses
    
    output_dir = args.output_dir or 'generated/hmrc_documentation'
    
    if not use_curses:
        # Run directly without curses
        scraper = HMRCScraper(output_dir)
        if args.discover_only:
            scraper.run_comprehensive_discovery(max_batches=max_batches)
            print(f"Discovered {len(scraper.discovered_urls)} HMRC documents")
        else:
            scraper.run_comprehensive_discovery(max_batches=max_batches)
            scraper.download_all_documents(args.max_documents)
            summary = scraper.generate_summary()
            scraper.create_training_datasets()
            
            print(f"\n=== HMRC DOCUMENTATION SCRAPING COMPLETE ===")
            print(f"Total documents discovered: {summary['total_discovered']}")
            print(f"Total documents downloaded: {summary['total_downloaded']}")
            print(f"Total failed downloads: {summary['total_failed']}")
    else:
        # Use the enhanced HMRC curses wrapper
        from utils.hmrc_curses_wrapper import run_hmrc_scraper_with_curses
        
        def hmrc_wrapper():
            scraper = HMRCScraper(output_dir)
            
            try:
                if args.discover_only:
                    scraper.run_comprehensive_discovery(max_batches=max_batches)
                    # Return result for discover-only mode
                    return {
                        'status': 'success',
                        'discovered': len(scraper.discovered_urls),
                        'downloaded': 0,
                        'failed': 0
                    }
                else:
                    scraper.run_comprehensive_discovery(max_batches=max_batches)
                    scraper.download_all_documents(args.max_documents)
                    summary = scraper.generate_summary()
                    scraper.create_training_datasets()
                    
                    # Return summary for the curses wrapper to know it completed
                    return {
                        'status': 'success',
                        'discovered': summary['total_discovered'],
                        'downloaded': summary['total_downloaded'],
                        'failed': summary['total_failed']
                    }
            except Exception as e:
                # Return error status
                return {
                    'status': 'error',
                    'error': str(e)
                }
        
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
                menu.add_item("No datasets found", "Create datasets using pipelines first", lambda x=None: x)
            
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
                        # Fix: Ensure current_dataset is not None before passing to function
                        if current_dataset:
                            metadata = manager.get_dataset_metadata(current_dataset)
                            print(f"\n=== Metadata for {current_dataset} ===")
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
                            with open(json_path, 'r') as f:
                                new_data = json.load(f)
                            # Fix: Ensure current_dataset is not None
                            if current_dataset:
                                manager.add_to_dataset(current_dataset, new_data)
                                print("Data added successfully!")
                        elif choice == "2":
                            print("Enter data as JSON (single object or array):")
                            json_str = input().strip()
                            # Fix: Import json module
                            new_data = json.loads(json_str)
                            # Fix: Ensure current_dataset is not None
                            if current_dataset:
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
                        # Fix: Ensure dataset_path construction is safe
                        if current_dataset:
                            dataset_path = manager.datasets_dir / current_dataset
                            dataset = manager.load_dataset(dataset_path)
                            print("Available fields:", dataset.column_names)
                            
                            print("\n1. Add new field")
                            print("2. Remove field")
                            print("3. Transform existing field")
                            choice = input("Select option (1-3): ").strip()
                            
                            if choice == "1":
                                field_name = input("Enter new field name: ").strip()
                                default_value = input("Enter default value for all rows: ").strip()
                                # Fix: Ensure current_dataset is not None
                                manager.add_dataset_field(current_dataset, field_name, 
                                                        lambda row: default_value)
                                print(f"Added field '{field_name}'")
                            elif choice == "2":
                                field_name = input("Enter field name to remove: ").strip()
                                # Fix: Ensure current_dataset is not None
                                manager.remove_dataset_field(current_dataset, field_name)
                                print(f"Removed field '{field_name}'")
                            elif choice == "3":
                                field_name = input("Enter field name to transform: ").strip()
                                print("Enter Python expression to transform (use 'x' for current value):")
                                transform_expr = input("Transform: ").strip()
                                transform_func = eval(f"lambda x: {transform_expr}")
                                # Fix: Ensure current_dataset is not None
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
                        # Fix: Import json module
                        json_str = input().strip()
                        metadata = json.loads(json_str)
                        # Fix: Ensure current_dataset is not None
                        if current_dataset:
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
                            # Fix: Ensure current_dataset is not None
                            if current_dataset:
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
                        # Fix: Ensure current_dataset is not None
                        if current_dataset:
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
                        # Try to import datasets with graceful fallback
                        try:
                            from datasets import Dataset
                        except ImportError:
                            print("Warning: 'datasets' module not found. Creating minimal dataset.")
                            
                            class Dataset:
                                @staticmethod
                                def from_list(data_list):
                                    return {"data": data_list}
                        
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

# Define missing helper functions
def _run_with_menu_args(func: Callable, description: str, **kwargs) -> Any:
    """Helper function to run a pipeline function from the menu with proper arguments"""
    print(f"Running {description}...")
    
    # Create a minimal args object with default values
    class Args:
        def __init__(self, **kwargs):
            self.output_dir = kwargs.get('output_dir', None)
            self.input_dir = kwargs.get('input_dir', None)
            self.max_documents = kwargs.get('max_documents', None)
            self.discover_only = kwargs.get('discover_only', False)
            self.resume = kwargs.get('resume', False)
            self.no_curses = kwargs.get('no_curses', False)
            self.url = kwargs.get('url', None)
            self.quick = kwargs.get('quick', False)
    
    # Add user input for common options
    print("\nOptions (press Enter for defaults):")
    output_dir = input("Output directory: ").strip() or kwargs.get('output_dir', None)
    max_docs = input("Max documents (Enter for all): ").strip()
    
    args = Args(
        output_dir=output_dir,
        max_documents=int(max_docs) if max_docs.isdigit() else None,
        **kwargs
    )
    
    try:
        result = func(args)
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

def _show_pipeline_status():
    """Show status of data directories and files"""
    print("\n=== PIPELINE STATUS ===")
    
    base_dir = Path("generated")
    if not base_dir.exists():
        print("No 'generated' directory found. No pipelines have been run yet.")
        return
    
    # List all subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print("No pipeline data found in the 'generated' directory.")
        return
    
    for subdir in sorted(subdirs):
        print(f"\n• {subdir.name}:")
        
        # Count files by type
        file_counts = {}
        total_size = 0
        
        for ext in ['.json', '.txt', '.html', '.md', '.csv', '.parquet']:
            files = list(subdir.glob(f"**/*{ext}"))
            if files:
                file_counts[ext] = len(files)
                total_size += sum(f.stat().st_size for f in files)
        
        if file_counts:
            for ext, count in file_counts.items():
                print(f"  - {count} {ext} files")
            
            # Show size in MB
            size_mb = total_size / (1024 * 1024)
            print(f"  - Total size: {size_mb:.1f} MB")
        else:
            print("  - No data files found")

def _show_documentation():
    """Show quick documentation and help"""
    print("\n=== Legal Llama Documentation ===")
    print("\nThis tool provides a unified interface for legal data collection and processing.")
    print("\nKey Components:")
    print("1. Data Collection Pipelines - Collect legal documents from various sources")
    print("2. Dataset Enhancement - Process and enhance legal texts for training")
    print("3. Training Optimization - Prepare datasets for Legal Llama training")
    
    print("\nUsage Examples:")
    print("• HMRC Tax Documentation: Collects tax guidance from the UK government")
    print("• Housing Legislation: Collects housing and tenancy legislation")
    print("• Case Law: Collects legal precedents from BAILII and other sources")
    
    print("\nFor more detailed documentation, see README.md")

def _manage_credentials():
    """Edit database and API credentials"""
    print("\n=== Credential Management ===")
    
    # Define credential locations
    env_file = Path(".env")
    
    # Define credentials to manage
    credential_definitions = {
        'MONGODB_CONNECTION_STRING': 'MongoDB Atlas connection string',
        'NEO4J_URI': 'Neo4j connection URI',
        'NEO4J_USERNAME': 'Neo4j username',
        'NEO4J_PASSWORD': 'Neo4j password',
        'PINECONE_API_KEY': 'Pinecone API key',
        'PINECONE_ENVIRONMENT': 'Pinecone environment',
        'ANTHROPIC_API_KEY': 'Anthropic API key for Claude integration',
        'OPENAI_API_KEY': 'OpenAI API key'
    }
    
    # Load current credentials
    current_values = {}
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    current_values[key] = value.strip('"\'')
    
    # List current credentials
    print("\nCurrent Credentials:")
    for key, description in credential_definitions.items():
        current_value = current_values.get(key, 'Not set')
        
        # Mask sensitive values for display
        if 'PASSWORD' in key or 'KEY' in key:
            display_value = '***Hidden***' if current_value != 'Not set' else 'Not set'
        else:
            display_value = current_value[:10] + '...' if len(current_value) > 10 else current_value
        
        print(f"{key}: {display_value} - {description}")
    
    # Edit credentials
    print("\nEnter credential key to edit (or 'q' to quit):")
    key = input("> ").strip().upper()
    
    if key.lower() == 'q':
        return
    
    if key in credential_definitions:
        new_value = input(f"Enter new value for {key} (leave empty to keep current): ").strip()
        if new_value:
            current_values[key] = new_value
            
            # Save credentials to .env file
            with open(env_file, 'w') as f:
                for k, v in current_values.items():
                    f.write(f"{k}={v}\n")
            
            print(f"Updated {key} successfully.")
        else:
            print("No changes made.")
    else:
        print(f"Unknown credential: {key}")
    
    # Ask if user wants to continue editing
    if input("\nEdit another credential? (y/N): ").strip().lower() == 'y':
        _manage_credentials()

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
                print("Exiting...")
                break
            
            # Convert to integer and call the corresponding function
            choice_num = int(choice)
            
            # Map the choice to the corresponding function
            if choice_num == 1:
                _run_with_menu_args(run_dynamic_pipeline, "Dynamic Pipeline")
            elif choice_num == 2:
                _run_with_menu_args(run_hmrc_scraper, "HMRC Scraper")
            elif choice_num == 3:
                _run_with_menu_args(run_housing_pipeline, "Housing Pipeline")
            elif choice_num == 4:
                _run_with_menu_args(run_bailii_scraper, "BAILII Scraper")
            elif choice_num == 5:
                _run_with_menu_args(run_copyright_pipeline, "Copyright Pipeline")
            elif choice_num == 6:
                _run_with_menu_args(run_complete_pipeline, "Complete Pipeline")
            elif choice_num == 7:
                _run_with_menu_args(run_legal_reasoning_enhancer, "Legal Reasoning Enhancer")
            elif choice_num == 8:
                _run_with_menu_args(run_tax_scenario_generator, "Tax Scenario Generator")
            elif choice_num == 9:
                _run_with_menu_args(run_advanced_qa_generator, "Advanced Q&A Generator")
            elif choice_num == 10:
                _run_with_menu_args(run_llama_training_optimizer, "Legal Llama Training Optimizer")
            elif choice_num == 11:
                _run_with_menu_args(run_enhanced_complete_pipeline, "Enhanced Complete Pipeline")
            elif choice_num == 12:
                _run_with_menu_args(run_enhanced_legal_pipeline, "Production Legal AI Pipeline")
            elif choice_num == 13:
                _run_with_menu_args(run_qa_generator, "Q&A Generator")
            elif choice_num == 14:
                _run_with_menu_args(run_database_ingestion, "Database Ingestion")
            elif choice_num == 15:
                _run_with_menu_args(run_dataset_manager, "Dataset Manager")
            elif choice_num == 16:
                _show_pipeline_status()
            elif choice_num == 17:
                _show_documentation()
            elif choice_num == 18:
                _manage_credentials()
            else:
                print("Invalid choice, please try again.")
                
        except ValueError:
            print("Please enter a number between 0 and 18.")
        except Exception as e:
            print(f"Error: {e}")
            
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Legal Llama Dataset Generation Framework")
    parser.add_argument('pipeline', nargs='?', help='Pipeline to run')
    parser.add_argument('--output-dir', help='Output directory for generated files')
    parser.add_argument('--input-dir', help='Input directory for source files')
    parser.add_argument('--max-documents', type=int, help='Maximum number of documents to process')
    parser.add_argument('--discover-only', action='store_true', help='Only discover documents, do not download')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    parser.add_argument('--no-curses', action='store_true', help='Disable curses interface')
    parser.add_argument('--quick', action='store_true', help='Run in quick mode (limited discovery)')
    parser.add_argument('--url', help='URL for dynamic pipeline')
    parser.add_argument('--version', action='store_true', help='Show version information and exit')
    
    args = parser.parse_args()
    
    if args.version:
        print_version_info()
        sys.exit(0)
    
    if not args.pipeline:
        # No pipeline specified - show interactive menu
        show_interactive_menu()
    else:
        # Run specified pipeline
        pipeline_map = {
            'dynamic': run_dynamic_pipeline,
            'hmrc': run_hmrc_scraper,
            'housing': run_housing_pipeline,
            'bailii': run_bailii_scraper,
            'copyright': run_copyright_pipeline,
            'complete': run_complete_pipeline,
            'legal-reasoning': run_legal_reasoning_enhancer,
            'tax-scenarios': run_tax_scenario_generator,
            'qa': run_qa_generator,
            'advanced-qa': run_advanced_qa_generator,
            'dataset-manager': run_dataset_manager,
            'llama-optimizer': run_llama_training_optimizer,
            'enhanced-complete': run_enhanced_complete_pipeline,
            'legal-ai': run_enhanced_legal_pipeline,
            'database': run_database_ingestion
        }
        
        if args.pipeline in pipeline_map:
            pipeline_map[args.pipeline](args)
        else:
            print(f"Unknown pipeline: {args.pipeline}")
            print("Available pipelines:", ", ".join(pipeline_map.keys()))
            sys.exit(1)