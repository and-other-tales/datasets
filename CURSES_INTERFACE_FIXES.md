# HMRC Curses Interface Fixes

## Overview

This document describes the improvements made to the HMRC scraper's curses terminal interface to resolve character encoding corruption and display issues.

## Problems Addressed

1. **Character Encoding Corruption**: Text in the curses interface was being corrupted, showing as symbols or unreadable characters.
2. **Screen Flickering**: The display was refreshing too frequently, causing visible flickering.
3. **Character Handling**: Unicode and special characters weren't being properly handled by the curses interface.
4. **Display Stability**: The interface would occasionally break under high load or when processing certain characters.

## Solutions Implemented

### 1. Improved Text Sanitization

- Enhanced the `_sanitize_text()` method to properly handle and replace problematic characters
- Added a comprehensive character replacement map for special Unicode characters
- Implemented stricter ASCII-only mode to ensure maximum terminal compatibility
- Added better fallback mechanisms when encoding fails

### 2. Optimized Screen Refreshing

- Increased the minimum interval between screen updates from 50ms to 100ms
- Reduced the number of messages processed at once from 10 to 5
- Replaced `clear()` with `erase()` for less flickering
- Used `noutrefresh()` and `doupdate()` instead of multiple `refresh()` calls for smoother updates

### 3. Better Window Management

- Added explicit coordinates when creating windows to avoid overlap
- Improved error handling for window operations
- Added proper terminal reset on exit
- Used `addnstr()` instead of `addstr()` to prevent buffer overflows

### 4. Enhanced Character Handling

- Added locale support for proper character encoding
- Set environment variable `NCURSES_NO_UTF8_ACS=1` to force ASCII mode
- Added multiple fallback mechanisms for rendering problematic characters
- Implemented a three-tiered approach to character display (attempt with color, without color, ASCII-only)

### 5. More Robust Error Handling

- Added try/except blocks around all curses operations
- Improved logging of curses-related errors
- Added graceful degradation when display operations fail

## Usage

To use the fixed implementation:

1. Import the fixed wrapper instead of the original one:
```python
from utils.hmrc_curses_wrapper_fixed import run_hmrc_scraper_with_curses
```

2. Use it the same way as before:
```python
run_hmrc_scraper_with_curses(your_scraper_function)
```

## Testing

A test script is provided to verify the fixes:
```bash
python test_fixed_hmrc_curses.py
```

This script runs a limited HMRC scraping operation with the fixed curses interface to demonstrate the improvements.

## Technical Notes

- The changes maintain backward compatibility with the original API
- No changes to the scraper pipeline or logic were required
- The fixes focus solely on display and character handling
- Performance impact is minimal (slight reduction in refresh rate by design)
