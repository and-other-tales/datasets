#!/usr/bin/env python3
"""
Enhanced Curses Wrapper for HMRC Scraper - Fixed Version
Provides proper log containment and sequential batch display
with improved character encoding and display handling
"""

import curses
import sys
import logging
import threading
import queue
import time
import re
import locale
import os
from collections import deque
from typing import Callable, Any, Dict, Optional, List

# Set locale to user's default for proper character encoding
locale.setlocale(locale.LC_ALL, '')

class HMRCCursesHandler(logging.Handler):
    """Custom logging handler specifically for HMRC scraper with rate limiting and improved character handling"""
    
    def __init__(self, log_window, max_lines=1000):
        super().__init__()
        self.log_window = log_window
        self.lock = threading.Lock()
        self.log_buffer = deque(maxlen=max_lines)
        self.batch_status = {}  # Track batch processing status
        self.last_update = 0
        self.update_interval = 0.1  # 100ms minimum between screen updates (increased from 50ms)
        self.pending_logs = queue.Queue()
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
    def emit(self, record):
        """Queue log records for processing"""
        try:
            msg = self.format(record)
            self.pending_logs.put(msg)
        except Exception:
            pass
    
    def process_logs(self):
        """Process queued logs with rate limiting"""
        messages_to_display = []
        
        # Collect up to 5 messages at once (reduced from 10 to prevent display flicker)
        for _ in range(5):
            try:
                msg = self.pending_logs.get_nowait()
                messages_to_display.append(msg)
            except queue.Empty:
                break
        
        if not messages_to_display:
            return
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            time.sleep(self.update_interval - (current_time - self.last_update))
        
        with self.lock:
            self.last_update = time.time()
            
            # Parse and organize batch messages
            for msg in messages_to_display:
                # Extract batch information if present
                batch_match = re.search(r'Batch (\d+)-(\d+):', msg)
                if batch_match:
                    batch_start = int(batch_match.group(1))
                    batch_end = int(batch_match.group(2))
                    self.batch_status[batch_start] = msg
                
                # Add to buffer
                self.log_buffer.append(msg)
            
            # Update display
            self._update_display()
    
    def _update_display(self):
        """Update the curses display with buffered logs - improved for character handling"""
        if self.log_window is None:
            return
        
        try:
            max_y, max_x = self.log_window.getmaxyx()
            if max_y <= 0 or max_x <= 0:
                return
            
            # Clear window with erase() instead of clear() for less flicker
            self.log_window.erase()
            
            # Calculate visible lines
            visible_lines = max_y - 1
            
            # Get the last N lines from buffer
            display_lines = list(self.log_buffer)[-visible_lines:]
            
            # Display lines
            for i, line in enumerate(display_lines):
                if i >= visible_lines:
                    break
                
                # Sanitize and truncate line
                safe_line = self._sanitize_text(line)
                if len(safe_line) > max_x - 1:
                    safe_line = safe_line[:max_x - 4] + "..."
                
                # Determine color based on content
                color = self._get_line_color(line)
                
                try:
                    # Try using addnstr which is safer for handling strings
                    self.log_window.addnstr(i, 0, safe_line, max_x - 1, color)
                except curses.error:
                    # Try without color if it fails
                    try:
                        self.log_window.addnstr(i, 0, safe_line, max_x - 1)
                    except curses.error:
                        # Last resort - simple ASCII only string
                        try:
                            ascii_line = ''.join(c if 32 <= ord(c) < 127 else '_' for c in safe_line)
                            self.log_window.addnstr(i, 0, ascii_line, max_x - 1)
                        except curses.error:
                            pass
            
            try:
                # Use noutrefresh() instead of refresh() for better performance
                self.log_window.noutrefresh()
                curses.doupdate()  # Update all windows at once
            except curses.error:
                pass
            
        except Exception:
            # Silently ignore display errors to keep the application running
            pass
    
    def _sanitize_text(self, text):
        """Sanitize text for curses display - improved to handle unicode properly"""
        if not text:
            return ""
        
        # First, convert to string if needed
        if not isinstance(text, str):
            text = str(text)
        
        # Replace problematic characters with simpler alternatives
        replacements = {
            '❌': 'X',  # Simplified characters
            '✅': 'V',
            '⚠': '!',
            '🔶': '*',
            '📊': '#',
            '🚀': '>',
            '📁': 'D',
            '💾': 'S',
            '–': '-',
            '—': '--',
            ''': "'",
            ''': "'",
            '"': '"',
            '"': '"',
            '…': '...',
            '•': '*',
            '■': '#',
            '►': '>',
            '●': 'o',
            '○': 'o',
            '◆': '*',
            '▪': '-',
            '▫': '-',
            '≤': '<=',
            '≥': '>=',
            '×': 'x',
            '÷': '/',
            '≠': '!=',
            '√': 'sqrt',
            '∞': 'inf',
            '≈': '~=',
            '∆': 'delta',
            '∑': 'sum',
            '∏': 'prod',
            '∂': 'd',
            '∫': 'int',
            '∴': '->',
            '∵': '<-',
            '∩': 'n',
            '∪': 'u',
            '⊂': 'c',
            '⊃': 'C',
            '⊆': 'c=',
            '⊇': 'C=',
            '∈': 'in',
            '∉': '!in',
            '∀': 'A',
            '∃': 'E',
            '∄': '!E',
            '∅': '{}'
        }
        
        safe_text = text
        for char, replacement in replacements.items():
            safe_text = safe_text.replace(char, replacement)
        
        # Approach 1: Keep ASCII only for maximum compatibility
        try:
            safe_text = ''.join(c if 32 <= ord(c) < 127 else '_' for c in safe_text)
        except Exception:
            # Fallback to a more aggressive replacement
            try:
                safe_text = safe_text.encode('ascii', errors='replace').decode('ascii')
            except Exception:
                # Last resort - remove all potentially problematic characters
                safe_text = ''.join(c for c in safe_text if 32 <= ord(c) < 127)
        
        return safe_text
    
    def _get_line_color(self, line):
        """Determine color based on log content"""
        if not hasattr(curses, 'color_pair'):
            return 0
        
        if "ERROR" in line or "Failed" in line:
            return curses.color_pair(4)  # Red
        elif "WARNING" in line:
            return curses.color_pair(3)  # Yellow
        elif "Successfully" in line or "completed" in line:
            return curses.color_pair(2)  # Green
        elif "Batch" in line and "Found" in line:
            return curses.color_pair(6)  # Cyan
        elif "API" in line:
            return curses.color_pair(5)  # Magenta
        else:
            return 0  # Default

class HMRCCursesWrapper:
    """Enhanced curses wrapper specifically for HMRC scraper - fixed version"""
    
    def __init__(self):
        self.stdscr = None
        self.header_win = None
        self.status_win = None
        self.log_win = None
        self.progress_win = None
        self.footer_win = None
        self.log_handler = None
        self.log_processor_thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()  # Initialize the lock
        
        # Store terminal dimensions as instance variables
        self.terminal_height = 0
        self.terminal_width = 0
        
        # Progress tracking
        self.total_discovered = 0
        self.total_downloaded = 0
        self.current_batch = 0
        self.batch_size = 100
        
    def run_with_curses(self, pipeline_func: Callable, *args, **kwargs):
        """Run the pipeline with enhanced curses interface"""
        try:
            # Force terminal to use ASCII mode for maximum compatibility
            os.environ.setdefault('NCURSES_NO_UTF8_ACS', '1')
            
            curses.wrapper(self._curses_main, pipeline_func, *args, **kwargs)
        except Exception as e:
            # Fallback to regular execution
            print(f"Curses interface error: {e}")
            print("Falling back to command-line execution...")
            try:
                result = pipeline_func(*args, **kwargs)
                print("[✓] HMRC Scraper completed successfully!")
            except Exception as pipeline_e:
                print(f"[X] HMRC Scraper failed: {pipeline_e}")
                sys.exit(1)
    
    def _curses_main(self, stdscr, pipeline_func: Callable, *args, **kwargs):
        """Main curses interface"""
        self.stdscr = stdscr
        
        # Initialize colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Header
        curses.init_pair(2, curses.COLOR_GREEN, -1)    # Success
        curses.init_pair(3, curses.COLOR_YELLOW, -1)   # Warning
        curses.init_pair(4, curses.COLOR_RED, -1)      # Error
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # API calls
        curses.init_pair(6, curses.COLOR_CYAN, -1)     # Batch info
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Footer
        
        # Hide cursor
        curses.curs_set(0)
        
        # Use raw mode to avoid character interpretation problems
        curses.raw()
        
        # Set non-blocking mode for input
        stdscr.nodelay(True)
        
        # Setup windows
        self._setup_windows()
        
        # Setup logging
        self._setup_logging()
        
        # Start log processor thread
        self.log_processor_thread = threading.Thread(
            target=self._log_processor_loop,
            daemon=True
        )
        self.log_processor_thread.start()
        
        # Run the pipeline in a separate thread
        pipeline_thread = threading.Thread(
            target=self._run_pipeline,
            args=(pipeline_func, args, kwargs),
            daemon=True
        )
        pipeline_thread.start()
        
        # Main event loop
        self._event_loop(pipeline_thread)
        
        # Cleanup
        self.stop_event.set()
        if self.log_processor_thread:
            self.log_processor_thread.join(timeout=1)
            
        # Reset terminal to normal mode        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()
        
    def _setup_windows(self):
        """Setup all curses windows with proper dimensions"""
        try:
            # Ensure stdscr is initialized
            if self.stdscr is None:
                raise RuntimeError("Standard screen not initialized")
            
            # Get terminal dimensions
            height, width = self.stdscr.getmaxyx()
            
            # Set these as instance variables so they're accessible in error handlers
            self.terminal_height = height
            self.terminal_width = width
            
            # Ensure minimum dimensions
            if height < 10 or width < 40:
                raise ValueError(f"Terminal size too small: {width}x{height}, minimum 40x10 required")
            
            # Window heights
            header_height = 4
            status_height = 4
            progress_height = 3
            footer_height = 2
            
            # Calculate log window height
            log_height = height - header_height - status_height - progress_height - footer_height - 2
            
            # Ensure log window has at least some space
            if log_height < 1:
                log_height = 1
            
            # Create windows with explicit coordinates to avoid overlap
            self.header_win = curses.newwin(header_height, width, 0, 0)
            self.status_win = curses.newwin(status_height, width, header_height, 0)
            self.progress_win = curses.newwin(progress_height, width, header_height + status_height, 0)
            self.log_win = curses.newwin(log_height, width, header_height + status_height + progress_height, 0)
            self.footer_win = curses.newwin(footer_height, width, height - footer_height, 0)
            
            # Enable scrolling for log window
            self.log_win.scrollok(True)
            self.log_win.idlok(True)
            
            # Clear all windows with erase() instead of clear()
            self.stdscr.erase()
            self.header_win.erase()
            self.status_win.erase()
            self.progress_win.erase()
            self.log_win.erase()
            self.footer_win.erase()
            
            # Draw static elements
            self._draw_header()
            self._draw_footer()
            self._update_status("Initializing...")
            self._update_progress()
            
            # Use doupdate() for more efficient screen updates
            self.stdscr.noutrefresh()
            curses.doupdate()
            
        except Exception as e:
            # If window setup fails, log the error and continue with minimal setup
            try:
                if self.stdscr is not None:
                    self.stdscr.erase()
                    # Use the instance variables we set earlier to avoid unbound errors
                    if self.terminal_width > 0:
                        self.stdscr.addnstr(0, 0, f"Window setup error: {e}", self.terminal_width - 1)
                        self.stdscr.addnstr(1, 0, "Continuing with simplified display...", self.terminal_width - 1)
                        self.stdscr.refresh()
                
                # Minimal windows - if stdscr is available, create a subwindow for logs
                if self.stdscr is not None and self.terminal_height > 4 and self.terminal_width > 0:
                    self.log_win = self.stdscr.subwin(self.terminal_height - 4, self.terminal_width, 2, 0)
                    if self.log_win is not None:
                        self.log_win.scrollok(True)
            except Exception:
                # If even this fails, we'll have no windows but the app will still run
                pass
    
    def _draw_header(self):
        """Draw the header"""
        if not self.header_win:
            return
        
        height, width = self.header_win.getmaxyx()
        self.header_win.erase()
        
        # Title
        title = "HMRC Documentation Scraper"
        subtitle = "othertales Datasets Tools"
        
        try:
            # Draw box
            self.header_win.box()
            
            # Center title
            if width > len(title) + 2:
                x = (width - len(title)) // 2
                self.header_win.addnstr(1, x, title, width - x - 1, curses.color_pair(1) | curses.A_BOLD)
            
            # Center subtitle
            if width > len(subtitle) + 2:
                x = (width - len(subtitle)) // 2
                self.header_win.addnstr(2, x, subtitle, width - x - 1, curses.color_pair(6))
            
        except curses.error:
            pass
        
        self.header_win.noutrefresh()
    
    def _draw_footer(self):
        """Draw the footer with controls"""
        if not self.footer_win:
            return
        
        height, width = self.footer_win.getmaxyx()
        self.footer_win.erase()
        
        controls = "Q: Quit | P: Pause/Resume | ↑↓: Scroll Logs"
        
        try:
            # Draw line separator with safer characters
            self.footer_win.addnstr(0, 0, "-" * (width - 1), width - 1, curses.color_pair(6))
            
            # Center controls
            if width > len(controls):
                x = (width - len(controls)) // 2
                self.footer_win.addnstr(1, x, controls, width - x - 1, curses.color_pair(7))
        except curses.error:
            pass
        
        self.footer_win.noutrefresh()
    
    def _update_status(self, status: str, is_error: bool = False):
        """Update the status window"""
        if not self.status_win:
            return
        
        height, width = self.status_win.getmaxyx()
        self.status_win.erase()
        
        try:
            # Draw box
            self.status_win.box()
            
            # Status label
            self.status_win.addnstr(1, 2, "Status: ", 8, curses.A_BOLD)
            
            # Status text - sanitize first
            status = self._sanitize_text(status)
            
            color = curses.color_pair(4) if is_error else curses.color_pair(2)
            max_status_len = width - 12
            if len(status) > max_status_len:
                status = status[:max_status_len - 3] + "..."
            
            self.status_win.addnstr(1, 10, status, max_status_len, color)
            
        except curses.error:
            pass
        
        self.status_win.noutrefresh()
    
    def _update_progress(self):
        """Update the progress window"""
        if not self.progress_win:
            return
        
        height, width = self.progress_win.getmaxyx()
        self.progress_win.erase()
        
        try:
            # Draw box
            self.progress_win.box()
            
            # Progress info
            progress_text = f"Discovered: {self.total_discovered} | Downloaded: {self.total_downloaded}"
            if self.current_batch > 0:
                progress_text += f" | Current Batch: {self.current_batch}-{self.current_batch + self.batch_size}"
            
            # Safe display
            if len(progress_text) < width - 4:
                self.progress_win.addnstr(1, 2, progress_text, width - 4, curses.color_pair(6))
            
        except curses.error:
            pass
        
        self.progress_win.noutrefresh()
    
    def _setup_logging(self):
        """Setup logging with custom handler"""
        # Create custom handler
        self.log_handler = HMRCCursesHandler(self.log_win)
        
        # Configure loggers
        loggers_to_configure = [
            logging.getLogger(),  # Root logger
            logging.getLogger('pipelines.hmrc_scraper'),
            logging.getLogger('hmrc_scraper'),
            logging.getLogger('__main__')
        ]
        
        for logger in loggers_to_configure:
            # Remove existing stream handlers
            handlers_to_remove = []
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handlers_to_remove.append(handler)
            
            for handler in handlers_to_remove:
                logger.removeHandler(handler)
            
            # Add our handler
            logger.addHandler(self.log_handler)
            logger.setLevel(logging.INFO)
    
    def _log_processor_loop(self):
        """Process logs in a separate thread"""
        while not self.stop_event.is_set():
            if self.log_handler:
                try:
                    self.log_handler.process_logs()
                except Exception:
                    # Ensure we don't crash the log processor
                    pass
            time.sleep(0.1)  # 100ms intervals (increased from 50ms)
    
    def _run_pipeline(self, pipeline_func: Callable, args, kwargs):
        """Run the pipeline function"""
        try:
            self._update_status("Running HMRC Scraper...")
            result = pipeline_func(*args, **kwargs)
            
            # Check result and update status accordingly
            if isinstance(result, dict):
                if result.get('status') == 'success':
                    discovered = result.get('discovered', 0)
                    downloaded = result.get('downloaded', 0)
                    self._update_status(f"Completed Successfully! Downloaded {downloaded}/{discovered} documents", is_error=False)
                elif result.get('status') == 'error':
                    self._update_status(f"Failed: {result.get('error', 'Unknown error')}", is_error=True)
                else:
                    self._update_status("Completed Successfully!", is_error=False)
            else:
                self._update_status("Completed Successfully!", is_error=False)
                
            # Trigger final update
            curses.doupdate()
        except Exception as e:
            import traceback
            error_msg = f"Failed: {str(e)}"            # Log the full traceback to file for debugging
            with open('hmrc_scraper_error.log', 'w') as f:
                f.write(traceback.format_exc())
            self._update_status(error_msg, is_error=True)
            
    def _event_loop(self, pipeline_thread):
        """Main event loop handling user input"""
        paused = False
        
        while pipeline_thread.is_alive() and self.log_handler and not self.log_handler.pending_logs.empty():
            try:
                # Check for user input
                if self.stdscr is None:
                    break
                    
                key = self.stdscr.getch()
                
                if key == ord('q') or key == ord('Q'):
                    # Quit
                    break
                elif key == ord('p') or key == ord('P'):
                    # Toggle pause
                    paused = not paused
                    status = "Paused" if paused else "Running..."
                    self._update_status(status)
                elif key == curses.KEY_UP:
                    # Scroll up (not implemented in this version)
                    pass
                elif key == curses.KEY_DOWN:
                    # Scroll down (not implemented in this version)
                    pass
                
                # Update progress periodically
                self._update_progress()
                
                # Use doupdate() for more efficient screen refresh
                curses.doupdate()
                
                # Small delay to prevent high CPU usage
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                break
            except Exception:
                # Handle any other exceptions gracefully
                break
        
        # Wait for user before exiting
        self._update_status("Press any key to exit...")
        if self.stdscr is not None:
            self.stdscr.nodelay(False)
            self.stdscr.getch()
    
    def _sanitize_text(self, text):
        """Utility method to sanitize text for display"""
        if self.log_handler:
            return self.log_handler._sanitize_text(text)
        
        # Fallback sanitization if handler not available
        if not text:
            return ""
        
        if not isinstance(text, str):
            text = str(text)
            
        # Keep ASCII only
        return ''.join(c if 32 <= ord(c) < 127 else '_' for c in text)

def run_hmrc_scraper_with_curses(scraper_func: Callable, *args, **kwargs):
    """Convenience function to run HMRC scraper with enhanced curses interface"""
    wrapper = HMRCCursesWrapper()
    wrapper.run_with_curses(scraper_func, *args, **kwargs)
