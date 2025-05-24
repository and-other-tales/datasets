import os
import re
import ast
import requests
from datetime import datetime
from pathlib import Path
from utils.version import get_version_string

# Constants
EULA_TEMPLATE_URL = "https://www.apple.com/legal/sla/docs/macOSSequoia.pdf"
MIT_LICENSE_URL = "https://raw.githubusercontent.com/github/choosealicense.com/gh-pages/_licenses/mit.txt"
GITHUB_API_BASE = "https://api.github.com/repos"

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
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Try multiple sources for GitHub URL
        possible_urls = [
            data['info'].get('project_urls', {}).get('Source', ''),
            data['info'].get('project_urls', {}).get('Homepage', ''),
            data['info'].get('project_urls', {}).get('Repository', ''),
            data['info'].get('home_page', ''),
            data['info'].get('download_url', ''),
        ]
        
        for repo_url in possible_urls:
            if repo_url and "github.com" in repo_url:
                # Clean up URL and extract owner/repo
                repo_url = repo_url.replace('https://', '').replace('http://', '')
                repo_url = repo_url.replace('www.', '').replace('.git', '')
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
        response = requests.get(url, headers=headers, timeout=5)
        if response.ok:
            data = response.json()
            license_type = data['license']['spdx_id']
            license_text = requests.get(data['download_url'], timeout=5).text
            return license_type, license_text
    except Exception:
        return None, None
    return None, None

def fetch_pypi_license(package_name):
    """Fetch license information directly from PyPI."""
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=5)
        response.raise_for_status()
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
            
        return license_type, license_text
        
    except Exception:
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
    print(f"\nFetching license information for {len(third_party_modules)} modules...")
    for i, module in enumerate(third_party_modules, 1):
        print(f"  [{i}/{len(third_party_modules)}] Checking {module}...")
        
        # First try GitHub
        repo = fetch_github_repo(module)
        license_name, license_text = None, None
        
        if repo:
            print(f"    Found GitHub repo: {repo}")
            license_name, license_text = fetch_github_license(repo)
            if license_name and license_text:
                print(f"    GitHub license found: {license_name}")
                modules_info[module] = (license_name, license_text, f"GitHub: {repo}")
            else:
                print(f"    GitHub license not accessible, trying PyPI...")
                # Fallback to PyPI
                license_name, license_text = fetch_pypi_license(module)
                if license_name and license_text:
                    print(f"    PyPI license found: {license_name}")
                    modules_info[module] = (license_name, license_text, f"PyPI package")
                else:
                    print(f"    No license information available")
        else:
            print(f"    No GitHub repository found, trying PyPI...")
            # Try PyPI directly
            license_name, license_text = fetch_pypi_license(module)
            if license_name and license_text:
                print(f"    PyPI license found: {license_name}")
                modules_info[module] = (license_name, license_text, f"PyPI package")
            else:
                print(f"    No license information available")
    
    print(f"\n{len(modules_info)} third-party components found with known licenses.")
    eula_text = build_eula(software_name, software_version, dev_name, dev_address, modules_info)

    with open("LICENSE.md", "w", encoding="utf-8") as f:
        f.write(eula_text)
    print("\nLicense information generated and saved as LICENSE.md")

if __name__ == "__main__":
    main()
