#!/usr/bin/env python3
"""
Push Script

This script:
1. Updates the VERSION file with the current timestamp-based build number
2. Runs mdgen.py to generate the LICENSE file
3. Performs git add, commit, and push operations
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add the current directory to the path so we can import from utils
sys.path.append(str(Path(__file__).parent))

from utils.version import write_version_file

def run_command(command):
    """Run a shell command and return the output"""
    print(f"Executing: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               text=True, capture_output=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def update_version():
    """Update VERSION file with timestamp-based build number"""
    try:
        version = write_version_file()
        print(f"VERSION file updated to: {version}")
        return True
    except Exception as e:
        print(f"Error updating VERSION file: {e}")
        return False

def run_mdgen():
    """Run mdgen.py to generate LICENSE and other files"""
    try:
        mdgen_path = Path(__file__).parent / "mdgen.py"
        if not mdgen_path.exists():
            print(f"Warning: mdgen.py not found at {mdgen_path}")
            return False
        
        command = f"python {mdgen_path}"
        return run_command(command)
    except Exception as e:
        print(f"Error running mdgen.py: {e}")
        return False

def git_operations(commit_message):
    """Perform git add, commit, and push operations"""
    # Check if we're in a git repository
    if not run_command("git rev-parse --is-inside-work-tree"):
        print("Error: Not inside a git repository")
        return False
    
    # Add all changes
    if not run_command("git add -A"):
        print("Error: git add failed")
        return False
    
    # Commit changes
    if not run_command(f'git commit -m "{commit_message}"'):
        print("Error: git commit failed")
        return False
    
    # Push changes
    if not run_command("git push"):
        print("Error: git push failed")
        return False
    
    return True

def main():
    """Main function to run the push script"""
    parser = argparse.ArgumentParser(description="Update version, generate files, and push to git")
    parser.add_argument('-m', '--message', required=True, 
                        help='Commit message for the git commit')
    parser.add_argument('--skip-version', action='store_true',
                        help='Skip updating the VERSION file')
    parser.add_argument('--skip-mdgen', action='store_true',
                        help='Skip running mdgen.py')
    parser.add_argument('--skip-push', action='store_true',
                        help='Skip git push operation')
    
    args = parser.parse_args()
    
    # Validate commit message
    if not args.message:
        print("Error: Commit message is required")
        return 1
    
    # Update VERSION file
    if not args.skip_version:
        if not update_version():
            print("Failed to update VERSION file")
            return 1
    
    # Run mdgen.py
    if not args.skip_mdgen:
        if not run_mdgen():
            print("Failed to run mdgen.py")
            return 1
    
    # Perform git operations
    if not args.skip_push:
        if not git_operations(args.message):
            print("Git operations failed")
            return 1
    
    print("All operations completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
