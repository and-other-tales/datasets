#!/usr/bin/env python3
"""
Curses Pipeline Wrapper
Provides consistent curses interface for all pipeline execution
"""

import curses
import sys
import logging
from pathlib import Path
from typing import Callable, Any, Dict, Optional

class CursesPipelineWrapper:
    """Wrapper to run pipelines with consistent curses interface"""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.stdscr = None
        self.status_window = None
        self.log_window = None
        self.footer_window = None
        
    def run_pipeline(self, pipeline_function: Callable, *args, **kwargs):
        """Run a pipeline function with curses interface"""
        try:
            curses.wrapper(self._curses_main, pipeline_function, *args, **kwargs)
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
            sys.exit(1)
    
    def _curses_main(self, stdscr, pipeline_function: Callable, *args, **kwargs):
        """Main curses interface"""
        self.stdscr = stdscr
        
        # Initialize colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)     # Header
        curses.init_pair(2, curses.COLOR_GREEN, -1)    # Success
        curses.init_pair(3, curses.COLOR_YELLOW, -1)   # Warning
        curses.init_pair(4, curses.COLOR_RED, -1)      # Error
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Footer
        
        # Hide cursor
        curses.curs_set(0)
        
        # Get screen dimensions
        height, width = stdscr.getmaxyx()
        
        # Create windows
        header_height = 3
        footer_height = 1
        status_height = 5
        log_height = height - header_height - footer_height - status_height - 2
        
        # Header window
        header_win = curses.newwin(header_height, width, 0, 0)
        
        # Status window
        self.status_window = curses.newwin(status_height, width, header_height, 0)
        
        # Log window 
        self.log_window = curses.newwin(log_height, width, header_height + status_height, 0)
        self.log_window.scrollok(True)
        
        # Footer window
        self.footer_window = curses.newwin(footer_height, width, height - footer_height, 0)
        
        # Draw header
        self._draw_header(header_win)
        
        # Draw initial status
        self._draw_status("Initializing...")
        
        # Draw footer
        self._draw_footer()
        
        # Setup logging to curses
        self._setup_curses_logging()
        
        # Refresh all windows
        header_win.refresh()
        self.status_window.refresh()
        self.log_window.refresh()
        self.footer_window.refresh()
        
        # Run the pipeline
        try:
            result = pipeline_function(*args, **kwargs)
            self._draw_status("Pipeline completed successfully", success=True)
            self.log_window.addstr(f"\n✅ {self.pipeline_name} completed successfully!\n")
            self.log_window.refresh()
        except Exception as e:
            self._draw_status(f"Pipeline failed: {e}", error=True)
            self.log_window.addstr(f"\n❌ {self.pipeline_name} failed: {e}\n")
            self.log_window.refresh()
        
        # Wait for key press
        self.log_window.addstr("\nPress any key to return to main menu...")
        self.log_window.refresh()
        stdscr.getch()
    
    def _draw_header(self, header_win):
        """Draw the header"""
        height, width = header_win.getmaxyx()
        
        # Clear header
        header_win.clear()
        
        # Title
        title = f"othertales Datasets Tools - {self.pipeline_name}"
        header_win.addstr(0, (width - len(title)) // 2, title, 
                         curses.color_pair(1) | curses.A_BOLD)
        
        # Border
        header_win.addstr(2, 0, "─" * width)
        
        header_win.refresh()
    
    def _draw_status(self, status: str, success: bool = False, error: bool = False):
        """Draw the status window"""
        if not self.status_window:
            return
        
        self.status_window.clear()
        
        # Status title
        self.status_window.addstr(0, 0, "Status:", curses.A_BOLD)
        
        # Status text with appropriate color
        color = curses.color_pair(2) if success else curses.color_pair(4) if error else curses.color_pair(3)
        self.status_window.addstr(1, 2, status, color)
        
        # Current phase (if available)
        # This would be updated by the PipelineController
        
        self.status_window.refresh()
    
    def _draw_footer(self):
        """Draw the footer with controls"""
        if not self.footer_window:
            return
        
        self.footer_window.clear()
        
        # Control text
        controls = "P:Pause/Resume | A:Update DB | D:Create Dataset | Q:Quit"
        width = self.footer_window.getmaxyx()[1]
        
        # Center the controls
        x = (width - len(controls)) // 2
        if x > 0:
            self.footer_window.addstr(0, x, controls, curses.color_pair(5))
        
        self.footer_window.refresh()
    
    def _setup_curses_logging(self):
        """Setup logging to display in curses window"""
        # Create a custom logging handler that writes to the curses window
        class CursesLogHandler(logging.Handler):
            def __init__(self, log_window):
                super().__init__()
                self.log_window = log_window
            
            def emit(self, record):
                try:
                    msg = self.format(record)
                    # Sanitize message for curses display
                    safe_msg = self._sanitize_text_for_curses(msg)
                    
                    # Check window size and handle scrolling
                    max_y, max_x = self.log_window.getmaxyx()
                    cur_y, cur_x = self.log_window.getyx()
                    
                    if cur_y >= max_y - 1:
                        self.log_window.scroll()
                        self.log_window.move(max_y - 2, 0)
                    
                    # Truncate if too long
                    if len(safe_msg) > max_x - 1:
                        safe_msg = safe_msg[:max_x - 4] + "..."
                    
                    self.log_window.addstr(f"{safe_msg}\n")
                    self.log_window.refresh()
                except Exception:
                    pass
            
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
        
        # Add the curses handler to the root logger
        curses_handler = CursesLogHandler(self.log_window)
        curses_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
        # Get the logger for our pipelines
        pipeline_logger = logging.getLogger()
        pipeline_logger.addHandler(curses_handler)
        pipeline_logger.setLevel(logging.INFO)

def run_pipeline_with_curses(pipeline_name: str, pipeline_function: Callable, *args, **kwargs):
    """Convenience function to run a pipeline with curses interface"""
    wrapper = CursesPipelineWrapper(pipeline_name)
    wrapper.run_pipeline(pipeline_function, *args, **kwargs)