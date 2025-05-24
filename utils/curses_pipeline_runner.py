#!/usr/bin/env python3
"""
Curses Pipeline Runner
Executes pipelines within curses interface with consistent styling
"""

import curses
import sys
import time
import threading
import queue
import logging
from pathlib import Path
from typing import Callable, Any, Dict, Optional, List
from io import StringIO

class CursesLogHandler(logging.Handler):
    """Custom logging handler that captures logs for curses display"""
    
    def __init__(self):
        super().__init__()
        self.log_queue = queue.Queue()
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            pass

class CursesPipelineRunner:
    """Runs pipelines within curses interface maintaining visual consistency"""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.stdscr = None
        self.header_win = None
        self.status_win = None
        self.log_win = None
        self.footer_win = None
        self.input_win = None
        
        self.current_phase = "Starting"
        self.current_status = "Initializing"
        self.is_paused = False
        self.log_handler = CursesLogHandler()
        
        # Dimensions
        self.header_height = 4
        self.status_height = 6
        self.footer_height = 2
        self.input_height = 8
    
    def run_pipeline_with_args(self, pipeline_func: Callable, args_collector: Callable = None):
        """Run pipeline with argument collection in curses"""
        try:
            curses.wrapper(self._main_interface, pipeline_func, args_collector)
        except KeyboardInterrupt:
            print("\nPipeline interrupted by user")
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
    
    def _main_interface(self, stdscr, pipeline_func: Callable, args_collector: Callable = None):
        """Main curses interface"""
        self.stdscr = stdscr
        
        # Initialize curses
        curses.start_color()
        curses.use_default_colors()
        
        # Color pairs (matching main menu style)
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Selected/Header
        curses.init_pair(2, curses.COLOR_CYAN, -1)    # Title/Categories  
        curses.init_pair(3, curses.COLOR_GREEN, -1)   # Success/Active
        curses.init_pair(4, curses.COLOR_YELLOW, -1)  # Warning/Info
        curses.init_pair(5, curses.COLOR_RED, -1)     # Error/Exit
        curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Footer
        
        curses.curs_set(0)  # Hide cursor initially
        
        # Setup windows
        self._setup_windows()
        
        # Setup logging
        self._setup_logging()
        
        # Collect arguments if needed
        args = None
        if args_collector:
            args = self._collect_arguments(args_collector)
            if args is None:  # User cancelled
                return
        
        # Run the pipeline
        self._run_pipeline(pipeline_func, args)
        
        # Show completion and wait for key
        self._show_completion()
    
    def _setup_windows(self):
        """Setup all curses windows"""
        height, width = self.stdscr.getmaxyx()
        
        # Calculate positions
        log_height = height - self.header_height - self.status_height - self.footer_height - 2
        
        # Create windows
        self.header_win = curses.newwin(self.header_height, width, 0, 0)
        self.status_win = curses.newwin(self.status_height, width, self.header_height, 0)
        self.log_win = curses.newwin(log_height, width, self.header_height + self.status_height, 0)
        self.footer_win = curses.newwin(self.footer_height, width, height - self.footer_height, 0)
        
        # Setup scrolling for log window
        self.log_win.scrollok(True)
        self.log_win.idlok(True)
        
        # Draw initial interface
        self._draw_header()
        self._draw_status()
        self._draw_footer()
        
        # Refresh all windows
        self.header_win.refresh()
        self.status_win.refresh() 
        self.log_win.refresh()
        self.footer_win.refresh()
    
    def _draw_header(self):
        """Draw header with pipeline name"""
        if not self.header_win:
            return
        
        height, width = self.header_win.getmaxyx()
        self.header_win.clear()
        
        # Main title
        title = "othertales Datasets Tools"
        self.header_win.addstr(0, (width - len(title)) // 2, title, curses.color_pair(2) | curses.A_BOLD)
        
        # Pipeline name
        subtitle = f"Running: {self.pipeline_name}"
        self.header_win.addstr(1, (width - len(subtitle)) // 2, subtitle, curses.color_pair(4))
        
        # Separator
        self.header_win.addstr(3, 0, "─" * width, curses.color_pair(2))
        
        self.header_win.refresh()
    
    def _draw_status(self):
        """Draw status window"""
        if not self.status_win:
            return
        
        height, width = self.status_win.getmaxyx()
        self.status_win.clear()
        
        # Status info
        self.status_win.addstr(0, 2, "Status:", curses.A_BOLD)
        
        # Current phase
        phase_text = f"Phase: {self.current_phase}"
        self.status_win.addstr(1, 4, phase_text, curses.color_pair(3))
        
        # Current status
        status_color = curses.color_pair(5) if self.is_paused else curses.color_pair(3)
        status_text = f"Status: {self.current_status}"
        self.status_win.addstr(2, 4, status_text, status_color)
        
        # Pipeline control reminder
        self.status_win.addstr(4, 2, "Controls: P=Pause/Resume, A=Update DB, D=Create Dataset, Q=Quit", 
                              curses.color_pair(4))
        
        self.status_win.refresh()
    
    def _draw_footer(self):
        """Draw footer (matching main menu style)"""
        if not self.footer_win:
            return
        
        height, width = self.footer_win.getmaxyx()
        self.footer_win.clear()
        
        # Control instructions
        controls = "↑↓: Scroll Log  |  P: Pause/Resume  |  Q: Return to Menu"
        
        # Center the controls
        x = (width - len(controls)) // 2
        if x > 0 and x + len(controls) < width:
            self.footer_win.addstr(0, x, controls, curses.color_pair(6))
        
        self.footer_win.refresh()
    
    def _setup_logging(self):
        """Setup logging to capture pipeline output"""
        # Add our handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.INFO)
        
        # Start log display thread
        self.log_thread = threading.Thread(target=self._log_display_thread, daemon=True)
        self.log_thread.start()
    
    def _sanitize_text_for_curses(self, text):
        """Sanitize text for safe curses display"""
        if not text:
            return ""
        
        # Replace problematic Unicode characters
        replacements = {
            '❌': '[X]',
            '✅': '[✓]', 
            '⚠': '[!]',
            '🔶': '[◆]',
            '📊': '[#]',
            '🚀': '[>]',
            '📁': '[D]',
            '💾': '[S]',
        }
        
        safe_text = text
        for emoji, replacement in replacements.items():
            safe_text = safe_text.replace(emoji, replacement)
        
        # Remove other non-ASCII characters that might cause issues
        try:
            safe_text = safe_text.encode('ascii', errors='replace').decode('ascii')
        except:
            safe_text = ''.join(c if ord(c) < 128 else '?' for c in safe_text)
        
        return safe_text
    
    def _log_display_thread(self):
        """Thread to display logs in curses window"""
        while True:
            try:
                # Get log message with timeout
                msg = self.log_handler.log_queue.get(timeout=0.1)
                
                # Add to log window
                if self.log_win:
                    # Determine color based on log level
                    color = curses.color_pair(3)  # Default green
                    if "ERROR" in msg:
                        color = curses.color_pair(5)  # Red
                    elif "WARNING" in msg:
                        color = curses.color_pair(4)  # Yellow
                    elif "🔶" in msg or "📊" in msg:
                        color = curses.color_pair(2)  # Cyan for status
                    
                    try:
                        # Sanitize message for curses display
                        safe_msg = self._sanitize_text_for_curses(msg)
                        
                        # Check if we have space in the window
                        max_y, max_x = self.log_win.getmaxyx()
                        cur_y, cur_x = self.log_win.getyx()
                        
                        if cur_y >= max_y - 1:
                            # Scroll window content up
                            self.log_win.scroll()
                            self.log_win.move(max_y - 2, 0)
                        
                        # Truncate message if too long for window
                        if len(safe_msg) > max_x - 1:
                            safe_msg = safe_msg[:max_x - 4] + "..."
                        
                        self.log_win.addstr(safe_msg + "\n", color)
                        self.log_win.refresh()
                    except curses.error:
                        # Window might be too small or character issues, ignore
                        pass
                        
            except queue.Empty:
                continue
            except Exception:
                break
    
    def _collect_arguments(self, args_collector: Callable):
        """Collect arguments within curses interface"""
        height, width = self.stdscr.getmaxyx()
        
        # Create input window
        input_win = curses.newwin(self.input_height, width - 4, 
                                 (height - self.input_height) // 2, 2)
        input_win.box()
        
        # Title
        title = f"Configure {self.pipeline_name}"
        input_win.addstr(0, (width - 4 - len(title)) // 2, f" {title} ", curses.color_pair(1))
        
        try:
            # Call the args collector with the input window
            args = args_collector(input_win)
            return args
        except Exception as e:
            # Show error and return None
            input_win.addstr(self.input_height - 2, 2, f"Error: {e}", curses.color_pair(5))
            input_win.addstr(self.input_height - 1, 2, "Press any key to continue...", curses.color_pair(4))
            input_win.refresh()
            input_win.getch()
            return None
        finally:
            del input_win
    
    def _run_pipeline(self, pipeline_func: Callable, args=None):
        """Run the pipeline function"""
        self.current_status = "Running"
        self._draw_status()
        
        try:
            if args:
                result = pipeline_func(args)
            else:
                result = pipeline_func()
            
            self.current_status = "Completed Successfully"
            self._draw_status()
            
        except Exception as e:
            self.current_status = f"Failed: {e}"
            self._draw_status()
            
            # Log the error
            if self.log_win:
                self.log_win.addstr(f"\n❌ Pipeline failed: {e}\n", curses.color_pair(5))
                self.log_win.refresh()
    
    def _show_completion(self):
        """Show completion message and wait for user"""
        if self.log_win:
            self.log_win.addstr(f"\n{'='*50}\n", curses.color_pair(2))
            self.log_win.addstr(f"Pipeline execution completed.\n", curses.color_pair(3))
            self.log_win.addstr(f"Press any key to return to main menu...\n", curses.color_pair(4))
            self.log_win.refresh()
        
        # Wait for key press
        self.stdscr.getch()

def run_pipeline_in_curses(pipeline_name: str, pipeline_func: Callable, args_collector: Callable = None):
    """Convenience function to run a pipeline in curses"""
    runner = CursesPipelineRunner(pipeline_name)
    runner.run_pipeline_with_args(pipeline_func, args_collector)