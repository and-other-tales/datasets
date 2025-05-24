#!/usr/bin/env python3
"""
HMRC Curses Interface Fix Utility
This script will replace the existing curses wrapper with the fixed version
throughout the project.
"""

import os
import sys
import shutil
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Apply curses interface fixes across the project"""
    
    # Get the base directory (project root)
    base_dir = Path(__file__).parent
    
    # Check if fixed version exists
    fixed_wrapper_path = base_dir / "utils" / "hmrc_curses_wrapper_fixed.py"
    original_wrapper_path = base_dir / "utils" / "hmrc_curses_wrapper.py"
    
    if not fixed_wrapper_path.exists():
        logger.error(f"Fixed wrapper not found at {fixed_wrapper_path}!")
        return False
    
    # Create backup of original file
    if original_wrapper_path.exists():
        backup_path = original_wrapper_path.with_suffix('.py.bak')
        logger.info(f"Creating backup of original wrapper at {backup_path}")
        shutil.copy2(original_wrapper_path, backup_path)
    
    # Replace the original with the fixed version
    logger.info(f"Replacing original wrapper with fixed version")
    shutil.copy2(fixed_wrapper_path, original_wrapper_path)
    
    # Update quick scraper import (just for safety - we already modified the file)
    quick_scraper_path = base_dir / "hmrc_quick_scraper.py"
    if quick_scraper_path.exists():
        logger.info(f"Updating imports in {quick_scraper_path}")
        with open(quick_scraper_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Ensure the import is from the original path (not fixed version)
        updated_content = content.replace(
            "from utils.hmrc_curses_wrapper_fixed import run_hmrc_scraper_with_curses",
            "from utils.hmrc_curses_wrapper import run_hmrc_scraper_with_curses"
        )
        
        if updated_content != content:
            with open(quick_scraper_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
    
    # Look for any test scripts that might use the wrapper
    for test_file in base_dir.glob("test_hmrc_curses*.py"):
        logger.info(f"Updating imports in {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update import to use the standard path
        updated_content = content.replace(
            "from utils.hmrc_curses_wrapper_fixed import run_hmrc_scraper_with_curses",
            "from utils.hmrc_curses_wrapper import run_hmrc_scraper_with_curses"
        )
        
        if updated_content != content:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
    
    logger.info("Curses interface fix has been applied successfully!")
    logger.info("You can now run HMRC scraper with the fixed curses interface")
    logger.info("If you encounter any issues, the original file is backed up at:")
    logger.info(f"  {backup_path}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
