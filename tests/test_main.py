#!/usr/bin/env python3
"""
Test the main entry point script
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_main_help():
    """Test that main.py shows help without errors"""
    import main
    
    # Override sys.argv to show help
    original_argv = sys.argv
    sys.argv = ['main.py', '--help']
    
    try:
        main.main()
    except SystemExit as e:
        # Help should exit with code 0
        assert e.code == 0
    finally:
        sys.argv = original_argv

def test_directory_structure():
    """Test that required directories exist"""
    required_dirs = ['pipelines', 'utils', 'generated', 'tests', 'logs']
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Directory {dir_name} should exist"
        assert dir_path.is_dir(), f"{dir_name} should be a directory"

def test_init_files():
    """Test that __init__.py files exist in packages"""
    package_dirs = ['pipelines', 'utils', 'tests']
    
    for dir_name in package_dirs:
        init_file = project_root / dir_name / '__init__.py'
        assert init_file.exists(), f"__init__.py should exist in {dir_name}"

if __name__ == "__main__":
    print("Running tests...")
    
    try:
        test_directory_structure()
        print("✓ Directory structure test passed")
        
        test_init_files()
        print("✓ Init files test passed")
        
        test_main_help()
        print("✓ Main help test passed")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)