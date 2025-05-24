#!/usr/bin/env python3
"""
Pipeline Controller for Pause/Resume Functionality
Handles keyboard input for pipeline control: P (pause/unpause), A (update databases), D (create dataset)
"""

import threading
import queue
import sys
import os
import select
import tty
import termios
import fcntl
import json
import logging
import curses
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class PipelineController:
    def __init__(self):
        self.is_paused = False
        self.is_running = True
        self.input_queue = queue.Queue()
        self.pause_point_data = {}
        self.callbacks = {}
        self.current_phase = "Starting"
        self.current_step = ""
        
        # Curses setup for footer display
        self.stdscr = None
        self.footer_displayed = False
        
        # Save original terminal settings
        self.old_settings = None
        if sys.stdin.isatty():
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
                # Set non-blocking input without breaking terminal output
                self._setup_non_blocking_input()
            except Exception:
                pass
        
        # Start input monitoring thread
        self.input_thread = threading.Thread(target=self._monitor_input, daemon=True)
        self.input_thread.start()
        
        # Initialize curses footer
        self._setup_curses_footer()
    
    def _setup_non_blocking_input(self):
        """Setup non-blocking input without breaking terminal output"""
        # Don't modify terminal settings at all during initialization
        # Only use select() for non-blocking reads
        pass
    
    def _monitor_input(self):
        """Monitor keyboard input in a separate thread"""
        while self.is_running:
            if sys.stdin.isatty():
                try:
                    # Use select with a short timeout to check for input
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if ready:
                        # Set terminal to raw mode temporarily to read single characters
                        old_settings = termios.tcgetattr(sys.stdin)
                        try:
                            tty.setraw(sys.stdin.fileno())
                            key = sys.stdin.read(1)
                            if key:
                                self.input_queue.put(key.lower())
                        finally:
                            # Always restore terminal settings
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass
            else:
                # Small delay to prevent busy waiting
                threading.Event().wait(0.1)
    
    def _restore_terminal(self):
        """Restore original terminal settings"""
        if sys.stdin.isatty() and self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass
    
    def register_callback(self, action: str, callback: Callable):
        """Register callback functions for actions"""
        self.callbacks[action] = callback
    
    def check_for_commands(self) -> Optional[str]:
        """Check for keyboard commands and handle them"""
        try:
            # Process all available keys
            commands_processed = []
            while not self.input_queue.empty():
                key = self.input_queue.get_nowait()
                
                if key == 'p':
                    self._handle_pause()
                    commands_processed.append('pause')
                elif key == 'a' and self.is_paused:
                    self._handle_database_update()
                    commands_processed.append('database_update')
                elif key == 'd' and self.is_paused:
                    self._handle_dataset_creation()
                    commands_processed.append('dataset_creation')
                elif key == 'q':
                    self._handle_quit()
                    commands_processed.append('quit')
            
            # Return the last processed command
            return commands_processed[-1] if commands_processed else None
                    
        except queue.Empty:
            pass
        return None
    
    def _handle_pause(self):
        """Handle pause/unpause functionality"""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            # Force output to be visible
            sys.stdout.write("\n" + "="*60 + "\n")
            sys.stdout.write("🔶 PIPELINE PAUSED\n")
            sys.stdout.write("="*60 + "\n")
            sys.stdout.write("Available commands:\n")
            sys.stdout.write("  P - Resume pipeline\n")
            sys.stdout.write("  A - Update databases from current point\n")
            sys.stdout.write("  D - Create dataset from current point\n")
            sys.stdout.write("  Q - Quit pipeline\n")
            sys.stdout.write("="*60 + "\n")
            sys.stdout.flush()
            
            # Save current pause point
            self.pause_point_data = {
                'timestamp': datetime.now().isoformat(),
                'status': 'paused',
                'current_phase': getattr(self, '_current_phase', 'unknown'),
                'progress': getattr(self, '_current_progress', {})
            }
            
            self._save_pause_state()
        else:
            sys.stdout.write("\n🔶 PIPELINE RESUMED\n")
            sys.stdout.write("Press P to pause again, Q to quit\n\n")
            sys.stdout.flush()
    
    def _handle_database_update(self):
        """Handle database update from current pause point"""
        print("\n🔄 UPDATING DATABASES FROM CURRENT POINT...")
        
        if 'database_update' in self.callbacks:
            try:
                self.callbacks['database_update'](self.pause_point_data)
                print("✅ Database update completed!")
            except Exception as e:
                print(f"❌ Database update failed: {e}")
                logger.error(f"Database update error: {e}")
        else:
            print("❌ Database update functionality not available")
        
        print("Press P to resume, D to create dataset, Q to quit")
    
    def _handle_dataset_creation(self):
        """Handle dataset creation from current pause point"""
        print("\n📊 CREATING DATASET FROM CURRENT POINT...")
        
        if 'dataset_creation' in self.callbacks:
            try:
                result = self.callbacks['dataset_creation'](self.pause_point_data)
                print(f"✅ Dataset creation completed! {result}")
            except Exception as e:
                print(f"❌ Dataset creation failed: {e}")
                logger.error(f"Dataset creation error: {e}")
        else:
            print("❌ Dataset creation functionality not available")
        
        print("Press P to resume, A to update databases, Q to quit")
    
    def _handle_quit(self):
        """Handle quit command"""
        print("\n🛑 QUITTING PIPELINE...")
        self.is_running = False
        self._restore_terminal()
        sys.exit(0)
    
    def _save_pause_state(self):
        """Save current pause state to file"""
        try:
            pause_file = Path('logs/pipeline_pause_state.json')
            pause_file.parent.mkdir(exist_ok=True)
            
            with open(pause_file, 'w') as f:
                json.dump(self.pause_point_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save pause state: {e}")
    
    def load_pause_state(self) -> Optional[Dict[str, Any]]:
        """Load previous pause state if exists"""
        try:
            pause_file = Path('logs/pipeline_pause_state.json')
            if pause_file.exists():
                with open(pause_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load pause state: {e}")
        return None
    
    def wait_while_paused(self):
        """Block execution while paused, but continue processing commands"""
        while self.is_paused and self.is_running:
            self.check_for_commands()
            threading.Event().wait(0.1)  # Small delay to prevent CPU spinning
    
    def _setup_curses_footer(self):
        """Setup curses footer for persistent display"""
        # For now, disable curses footer to avoid conflicts with normal terminal output
        # The main menu already provides curses interface
        self.footer_displayed = False
        logger.info("🔶 Pipeline Control: Press P to pause/resume, A to update databases (when paused), D to create dataset (when paused), Q to quit")
    
    def _update_footer(self):
        """Update the footer display"""
        # Display phase and status updates via normal logging
        status = "PAUSED" if self.is_paused else "RUNNING"
        phase_info = f"Phase: {self.current_phase}"
        if self.current_step:
            phase_info += f" - {self.current_step}"
        
        logger.info(f"📊 {phase_info} | Status: {status}")
    
    def set_current_phase(self, phase: str, progress: Dict[str, Any] = None):
        """Set current pipeline phase for pause state tracking"""
        self._current_phase = phase
        self._current_progress = progress or {}
        self.current_phase = phase
        self.current_step = progress.get('step', '') if progress else ''
        self._update_footer()
    
    def cleanup(self):
        """Cleanup controller resources"""
        self.is_running = False
        self._restore_terminal()
        
        # Wait for input thread to finish
        if hasattr(self, 'input_thread') and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)

def create_database_update_callback(pipeline_instance):
    """Create a database update callback for a pipeline"""
    def update_databases(pause_data):
        # Import here to avoid circular imports
        from utils.multi_database_ingestion import LegalDataIngester
        
        # Get database configuration
        import os
        config = {
            'mongodb': {
                'connection_string': os.getenv('MONGODB_CONNECTION_STRING', 'mongodb://localhost:27017/'),
                'database': os.getenv('MONGODB_DATABASE', 'legal_datasets')
            },
            'neo4j': {
                'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
                'password': os.getenv('NEO4J_PASSWORD', 'password')
            },
            'pinecone': {
                'api_key': os.getenv('PINECONE_API_KEY'),
                'environment': os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
            }
        }
        
        # Create ingester and update databases
        ingester = LegalDataIngester(config)
        ingester.ingest_from_directory(pipeline_instance.output_dir)
        ingester.close_connections()
        
        return "Database update completed successfully"
    
    return update_databases

def create_dataset_creation_callback(pipeline_instance):
    """Create a dataset creation callback for a pipeline"""
    def create_dataset(pause_data):
        # Use the pipeline's dataset creator
        if hasattr(pipeline_instance, 'dataset_creator'):
            datasets = pipeline_instance.dataset_creator.create_all_datasets()
            total_examples = sum(len(dataset) for dataset in datasets.values())
            return f"Created {len(datasets)} datasets with {total_examples} examples"
        else:
            # Fallback: create basic dataset from current data
            from utils.dataset_creator import UKLegislationDatasetCreator
            creator = UKLegislationDatasetCreator(
                source_dir=pipeline_instance.output_dir,
                output_dir=getattr(pipeline_instance, 'dataset_dir', 'generated/datasets')
            )
            datasets = creator.create_all_datasets()
            total_examples = sum(len(dataset) for dataset in datasets.values())
            return f"Created {len(datasets)} datasets with {total_examples} examples"
    
    return create_dataset