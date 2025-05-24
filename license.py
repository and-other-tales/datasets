import os
import re
import ast
import requests
import time
from datetime import datetime
from pathlib import Path
from utils.version import get_version_string

# Constants
EULA_TEMPLATE_URL = "https://www.apple.com/legal/sla/docs/macOSSequoia.pdf"
MIT_LICENSE_URL = "https://raw.githubusercontent.com/github/choosealicense.com/gh-pages/_licenses/mit.txt"
GITHUB_API_BASE = "https://api.github.com/repos"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# License verification mappings
LICENSE_MAPPINGS = {
    'Apache Software License': 'Apache-2.0',
    'Apache License': 'Apache-2.0',
    'MIT License': 'MIT',
    'BSD License': 'BSD-3-Clause',
    'GNU General Public License (GPL)': 'GPL-3.0',
    'GNU Lesser General Public License v3 or later (LGPLv3+)': 'LGPL-3.0+',
    'Python Software Foundation License': 'PSF-2.0',
    'Mozilla Public License 2.0 (MPL 2.0)': 'MPL-2.0',
    'ISC License (ISCL)': 'ISC',
    'Zope Public License': 'ZPL-2.1'
}

def make_request_with_retry(url, headers=None, timeout=5):
    """Make HTTP request with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 429:  # Rate limited
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                print(f"    Rate limited, waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise e
            wait_time = RETRY_DELAY * (2 ** attempt)
            print(f"    Request failed, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)
    return None

def normalize_license_name(license_name):
    """Normalize license names for consistency."""
    if not license_name or license_name.lower() in ['unknown', 'none', '']:
        return None
    
    # Check for exact mapping
    if license_name in LICENSE_MAPPINGS:
        return LICENSE_MAPPINGS[license_name]
    
    # Fuzzy matching for common variations
    license_lower = license_name.lower()
    
    if 'apache' in license_lower:
        if '2.0' in license_lower or 'apache-2.0' in license_lower:
            return 'Apache-2.0'
        return 'Apache-2.0'  # Default to 2.0
    
    if 'mit' in license_lower:
        return 'MIT'
    
    if 'bsd' in license_lower:
        if '2-clause' in license_lower:
            return 'BSD-2-Clause'
        elif '3-clause' in license_lower:
            return 'BSD-3-Clause'
        return 'BSD-3-Clause'  # Default to 3-clause
    
    if 'gpl' in license_lower:
        if 'lgpl' in license_lower or 'lesser' in license_lower:
            return 'LGPL-3.0+'
        if 'v3' in license_lower or '3.0' in license_lower:
            return 'GPL-3.0'
        if 'v2' in license_lower or '2.0' in license_lower:
            return 'GPL-2.0'
        return 'GPL-3.0'  # Default to v3
    
    if 'mozilla' in license_lower or 'mpl' in license_lower:
        return 'MPL-2.0'
    
    if 'isc' in license_lower:
        return 'ISC'
    
    return license_name  # Return original if no mapping found

def verify_license_consistency(github_license, pypi_license):
    """Verify consistency between GitHub and PyPI license information."""
    if not github_license or not pypi_license:
        return True  # Can't verify if one is missing
    
    github_normalized = normalize_license_name(github_license)
    pypi_normalized = normalize_license_name(pypi_license)
    
    if github_normalized == pypi_normalized:
        return True
    
    # Check for common aliases
    aliases = {
        'Apache-2.0': ['Apache Software License', 'Apache License'],
        'MIT': ['MIT License'],
        'BSD-3-Clause': ['BSD License', 'BSD'],
        'GPL-3.0': ['GNU General Public License (GPL)', 'GPL'],
        'PSF-2.0': ['Python Software Foundation License']
    }
    
    for canonical, alias_list in aliases.items():
        if ((github_normalized == canonical and pypi_license in alias_list) or
            (pypi_normalized == canonical and github_license in alias_list)):
            return True
    
    return False

def prompt_user():
    print("Please enter the following details for the EULA:")
    software_name = input("Software Name: ").strip()
    
    # Get dynamic version from git
    dynamic_version = get_version_string()
    software_version = input(f"Software Version [{dynamic_version}]: ").strip()
    if not software_version:
        software_version = dynamic_version
        
    developer_name = input("Developer Name [Adventures of the Persistently Impaired (...and Other Tales) Limited]: ").strip()
    if not developer_name:
        developer_name = "Adventures of the Persistently Impaired (...and Other Tales) Limited"
    developer_address = input("Developer Address [85 Great Portland Street, London W1W 7LT, United Kingdom]: ").strip()
    if not developer_address:
        developer_address = "85 Great Portland Street, London W1W 7LT, United Kingdom"
    return software_name, software_version, developer_name, developer_address

def extract_imports_from_code(base_dir):
    imports = set()
    file_count = 0
    for root, dirs, files in os.walk(base_dir):
        # Skip virtual environment directories
        dirs[:] = [d for d in dirs if d not in ['.venv', 'venv', '__pycache__', '.git']]
        
        for file in files:
            if file.endswith(".py"):
                file_count += 1
                file_path = os.path.join(root, file)
                print(f"  Scanning {file_path}...")
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for n in node.names:
                                    imports.add(n.name.split('.')[0])
                            elif isinstance(node, ast.ImportFrom) and node.module:
                                imports.add(node.module.split('.')[0])
                    except SyntaxError:
                        print(f"    Warning: Skipping {file_path} (syntax error)")
    print(f"  Scanned {file_count} Python files, found {len(imports)} unique imports")
    return sorted(imports)

def fetch_github_repo(package_name):
    try:
        response = make_request_with_retry(f"https://pypi.org/pypi/{package_name}/json")
        data = response.json()
        
        # Try multiple sources for GitHub URL
        possible_urls = [
            data['info'].get('project_urls', {}).get('Source', ''),
            data['info'].get('project_urls', {}).get('Homepage', ''),
            data['info'].get('project_urls', {}).get('Repository', ''),
            data['info'].get('project_urls', {}).get('Bug Tracker', ''),
            data['info'].get('home_page', ''),
            data['info'].get('download_url', ''),
        ]
        
        for repo_url in possible_urls:
            if repo_url and "github.com" in repo_url:
                # Clean up URL and extract owner/repo
                repo_url = repo_url.replace('https://', '').replace('http://', '')
                repo_url = repo_url.replace('www.', '').replace('.git', '')
                repo_url = repo_url.split('#')[0].split('?')[0]  # Remove anchors and query params
                parts = repo_url.split("github.com/")[-1].split("/")
                if len(parts) >= 2:
                    owner, repo = parts[0], parts[1]
                    return f"{owner}/{repo}".strip("/")
    except Exception:
        return None
    return None

def fetch_github_license(repo):
    try:
        url = f"{GITHUB_API_BASE}/{repo}/license"
        headers = {'Accept': 'application/vnd.github.v3+json'}
        response = make_request_with_retry(url, headers=headers)
        
        if response and response.ok:
            data = response.json()
            license_type = data['license']['spdx_id']
            # Get license text with retry
            license_response = make_request_with_retry(data['download_url'])
            if license_response:
                license_text = license_response.text
                return normalize_license_name(license_type), license_text
    except Exception as e:
        print(f"    GitHub API error: {str(e)}")
        return None, None
    return None, None

def fetch_pypi_license(package_name):
    """Fetch license information directly from PyPI."""
    try:
        response = make_request_with_retry(f"https://pypi.org/pypi/{package_name}/json")
        data = response.json()
        
        info = data['info']
        license_text = info.get('license', '')
        
        # Extract license type from classifiers
        license_type = 'Unknown'
        classifiers = info.get('classifiers', [])
        for classifier in classifiers:
            if classifier.startswith('License ::'):
                # Extract license name from classifier
                parts = classifier.split(' :: ')
                if len(parts) >= 3:
                    license_type = parts[2]
                    break
        
        # If no license text but have classifier, use classifier
        if not license_text.strip() and license_type != 'Unknown':
            license_text = f"Licensed under {license_type}"
        
        # Skip if no useful license information
        if not license_text.strip() or license_text.strip().upper() in ['UNKNOWN', 'NONE', '']:
            return None, None
        
        # Normalize license type    
        normalized_type = normalize_license_name(license_type)
        if not normalized_type:
            return None, None
            
        return normalized_type, license_text
        
    except Exception as e:
        print(f"    PyPI API error: {str(e)}")
        return None, None

def build_eula(software_name, software_version, developer_name, developer_address, modules_info):
    current_year = datetime.now().year
    eula = f"""
END USER LICENSE AGREEMENT FOR {software_name.upper()}

IMPORTANT - READ CAREFULLY: This {software_name} End User License Agreement ("EULA") is a legal agreement between you and {developer_name}.

COPYRIGHT (C) {current_year} {developer_name}, {developer_address}
All rights reserved.

This software is licensed, not sold. You may not copy, modify, reverse-engineer, decompile, or distribute this software without prior written consent.

This software may include third-party open source components. The licenses and attributions for such components are listed below.

=========================
Open Source Components
=========================
"""
    for mod, (license_name, license_text, copyright) in modules_info.items():
        if license_name and license_text:
            eula += f"\nComponent: {mod}\nLicense: {license_name}\n\n"
            eula += license_text
            eula += "\n" + "="*50 + "\n"
    
    # Add version at the end
    eula += f"\nVersion: {software_version}\n"
    return eula

def main():
    software_name, software_version, dev_name, dev_address = prompt_user()
    print("\nScanning source files and extracting imports...")
    third_party_modules = extract_imports_from_code(".")

    modules_info = {}
    verification_warnings = []
    print(f"\nFetching license information for {len(third_party_modules)} modules...")
    
    for i, module in enumerate(third_party_modules, 1):
        print(f"  [{i}/{len(third_party_modules)}] Checking {module}...")
        
        # Try both GitHub and PyPI for cross-validation
        repo = fetch_github_repo(module)
        github_license_name, github_license_text = None, None
        pypi_license_name, pypi_license_text = None, None
        
        # Always try PyPI first (more reliable)
        pypi_license_name, pypi_license_text = fetch_pypi_license(module)
        
        if repo:
            print(f"    Found GitHub repo: {repo}")
            github_license_name, github_license_text = fetch_github_license(repo)
            
            # Cross-validation: compare GitHub and PyPI licenses
            if github_license_name and pypi_license_name:
                is_consistent = verify_license_consistency(github_license_name, pypi_license_name)
                if is_consistent:
                    print(f"    ✓ Verified license: {github_license_name} (GitHub + PyPI consistent)")
                    # Prefer GitHub license text if available and longer
                    if github_license_text and len(github_license_text) > len(pypi_license_text or ''):
                        modules_info[module] = (github_license_name, github_license_text, f"GitHub: {repo} (verified)")
                    else:
                        modules_info[module] = (pypi_license_name, pypi_license_text, f"PyPI (GitHub verified)")
                else:
                    warning = f"License mismatch for {module}: GitHub={github_license_name}, PyPI={pypi_license_name}"
                    verification_warnings.append(warning)
                    print(f"    ⚠ License mismatch: GitHub={github_license_name}, PyPI={pypi_license_name}")
                    # Use PyPI as it's generally more reliable
                    if pypi_license_name and pypi_license_text:
                        modules_info[module] = (pypi_license_name, pypi_license_text, f"PyPI (GitHub conflict)")
                    else:
                        modules_info[module] = (github_license_name, github_license_text, f"GitHub: {repo} (unverified)")
            elif github_license_name and github_license_text:
                print(f"    GitHub license found: {github_license_name} (PyPI unavailable)")
                modules_info[module] = (github_license_name, github_license_text, f"GitHub: {repo}")
            elif pypi_license_name and pypi_license_text:
                print(f"    PyPI license found: {pypi_license_name} (GitHub inaccessible)")
                modules_info[module] = (pypi_license_name, pypi_license_text, f"PyPI package")
            else:
                print(f"    No license information available from either source")
        else:
            print(f"    No GitHub repository found")
            if pypi_license_name and pypi_license_text:
                print(f"    PyPI license found: {pypi_license_name}")
                modules_info[module] = (pypi_license_name, pypi_license_text, f"PyPI package")
            else:
                print(f"    No license information available")
    
    # Report verification warnings
    if verification_warnings:
        print(f"\n⚠ License verification warnings:")
        for warning in verification_warnings:
            print(f"  - {warning}")
    
    print(f"\n{len(modules_info)} third-party components found with known licenses.")
    eula_text = build_eula(software_name, software_version, dev_name, dev_address, modules_info)

    with open("LICENSE.md", "w", encoding="utf-8") as f:
        f.write(eula_text)
    print("\nLicense information generated and saved as LICENSE.md")

if __name__ == "__main__":
    main()
