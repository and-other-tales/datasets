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
            # Check if it's a curses-specific error
            error_str = str(e)
            curses_errors = [
                "addwstr() returned ERR",
                "addstr() returned ERR", 
                "nocbreak() returned ERR",
                "nodelay() returned ERR",
                "keypad() returned ERR",
                "curs_set() returned ERR"
            ]
            
            if any(err in error_str for err in curses_errors):
                print(f"Curses interface error: {e}")
                print("Falling back to command-line execution...")
                # Run without curses interface
                try:
                    result = pipeline_function(*args, **kwargs)
                    print(f"[✓] {self.pipeline_name} completed successfully!")
                except Exception as pipeline_e:
                    print(f"[X] {self.pipeline_name} failed: {pipeline_e}")
                    sys.exit(1)
            else:
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
            try:
                self.log_window.addstr(f"\n[✓] {self.pipeline_name} completed successfully!\n")
                self.log_window.refresh()
            except curses.error:
                pass
        except Exception as e:
            self._draw_status(f"Pipeline failed: {e}", error=True)
            try:
                self.log_window.addstr(f"\n[X] {self.pipeline_name} failed: {e}\n")
                self.log_window.refresh()
            except curses.error:
                pass
        
        # Wait for key press
        try:
            self.log_window.addstr("\nPress any key to return to main menu...")
            self.log_window.refresh()
            stdscr.getch()
        except curses.error:
            # If curses fails, just wait briefly and return
            import time
            time.sleep(2)
    
    def _draw_header(self, header_win):
        """Draw the header with bounds checking"""
        height, width = header_win.getmaxyx()
        
        # Clear header
        header_win.clear()
        
        # Title with bounds checking
        title = f"othertales Datasets Tools - {self.pipeline_name}"
        if len(title) > width - 2:
            title = title[:width - 5] + "..."
        
        try:
            x = max(0, (width - len(title)) // 2)
            if x + len(title) < width and height > 0:
                header_win.addstr(0, x, title, curses.color_pair(1) | curses.A_BOLD)
        except curses.error:
            # Fallback without color/formatting
            try:
                x = max(0, (width - len(title)) // 2)
                if x + len(title) < width and height > 0:
                    header_win.addstr(0, x, title)
            except curses.error:
                pass
        
        # Border with bounds checking
        try:
            if height > 2 and width > 0:
                border = "-" * min(width - 1, width)
                header_win.addstr(2, 0, border)
        except curses.error:
            pass
        
        header_win.refresh()
    
    def _draw_status(self, status: str, success: bool = False, error: bool = False):
        """Draw the status window with bounds checking"""
        if not self.status_window:
            return
        
        height, width = self.status_window.getmaxyx()
        self.status_window.clear()
        
        # Status title with bounds checking
        try:
            if height > 0 and width > 8:
                self.status_window.addstr(0, 0, "Status:", curses.A_BOLD)
        except curses.error:
            try:
                if height > 0 and width > 8:
                    self.status_window.addstr(0, 0, "Status:")
            except curses.error:
                pass
        
        # Status text with appropriate color and bounds checking
        if status and len(status) > width - 4:
            status = status[:width - 7] + "..."
        
        try:
            if height > 1 and width > 4 and status:
                color = curses.color_pair(2) if success else curses.color_pair(4) if error else curses.color_pair(3)
                self.status_window.addstr(1, 2, status, color)
        except curses.error:
            try:
                if height > 1 and width > 4 and status:
                    self.status_window.addstr(1, 2, status)
            except curses.error:
                pass
        
        self.status_window.refresh()
    
    def _draw_footer(self):
        """Draw the footer with controls and bounds checking"""
        if not self.footer_window:
            return
        
        height, width = self.footer_window.getmaxyx()
        self.footer_window.clear()
        
        # Control text with bounds checking
        controls = "P:Pause/Resume | A:Update DB | D:Create Dataset | Q:Quit"
        if len(controls) > width - 2:
            controls = "P:Pause | A:Update | D:Dataset | Q:Quit"
        if len(controls) > width - 2:
            controls = "P:Pause | Q:Quit"
        
        # Center the controls with bounds checking
        if height > 0 and width > len(controls):
            x = max(0, (width - len(controls)) // 2)
            try:
                if x + len(controls) < width:
                    self.footer_window.addstr(0, x, controls, curses.color_pair(5))
            except curses.error:
                try:
                    if x + len(controls) < width:
                        self.footer_window.addstr(0, x, controls)
                except curses.error:
                    pass
        
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
                    if max_y <= 0 or max_x <= 0:
                        return
                    
                    cur_y, cur_x = self.log_window.getyx()
                    
                    if cur_y >= max_y - 1:
                        self.log_window.scroll()
                        self.log_window.move(max(0, max_y - 2), 0)
                    
                    # Truncate if too long
                    if len(safe_msg) > max_x - 1:
                        safe_msg = safe_msg[:max(1, max_x - 4)] + "..."
                    
                    # Bounds check before adding string
                    current_y, current_x = self.log_window.getyx()
                    if current_y < max_y - 1 and len(safe_msg) > 0:
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