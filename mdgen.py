import os
import re
import ast
import sys
import requests
import time
import json
import importlib.metadata
import importlib.util
from datetime import datetime
from pathlib import Path
from utils.version import get_version_string

# Constants
EULA_TEMPLATE_URL = "https://www.apple.com/legal/sla/docs/macOSSequoia.pdf"
MIT_LICENSE_URL = "https://raw.githubusercontent.com/github/choosealicense.com/gh-pages/_licenses/mit.txt"
GITHUB_API_BASE = "https://api.github.com/repos"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Common license file names to check
LICENSE_FILE_NAMES = [
    'LICENSE', 'LICENSE.txt', 'LICENSE.md', 'LICENSE.rst',
    'LICENCE', 'LICENCE.txt', 'LICENCE.md', 'LICENCE.rst',
    'COPYING', 'COPYING.txt', 'COPYING.md',
    'COPYRIGHT', 'COPYRIGHT.txt', 'COPYRIGHT.md',
    'license', 'licence', 'copying', 'copyright'
]

# Cache for license lookups to avoid repeated API calls
license_cache = {}

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

def get_local_modules(base_dir):
    """Get all local module names from the project."""
    local_modules = set()
    
    # Get all Python files in root directory (without .py extension)
    for file in os.listdir(base_dir):
        if file.endswith('.py') and file != '__init__.py':
            module_name = file[:-3]  # Remove .py extension
            local_modules.add(module_name)
    
    # Get all directories with __init__.py (packages) or any .py files
    for root, dirs, files in os.walk(base_dir):
        # Skip virtual environment and special directories
        dirs[:] = [d for d in dirs if d not in ['.venv', 'venv', '__pycache__', '.git', 'node_modules', '.tox']]
        
        # Check if directory contains Python files
        has_py_files = any(f.endswith('.py') for f in files)
        
        if '__init__.py' in files or has_py_files:
            # Convert directory path to module name
            rel_path = os.path.relpath(root, base_dir)
            if rel_path != '.':
                module_name = rel_path.replace(os.path.sep, '.')
                local_modules.add(module_name)
                # Also add parent modules
                parts = module_name.split('.')
                for i in range(len(parts)):
                    local_modules.add('.'.join(parts[:i+1]))
    
    return local_modules

def extract_imports_from_code(base_dir):
    imports = set()
    file_count = 0
    
    # First, get all local modules
    local_modules = get_local_modules(base_dir)
    print(f"  Found {len(local_modules)} local modules: {', '.join(sorted(local_modules))}")
    
    for root, dirs, files in os.walk(base_dir):
        # Skip virtual environment directories
        dirs[:] = [d for d in dirs if d not in ['.venv', 'venv', '__pycache__', '.git', 'node_modules', '.tox']]
        
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
                                    module_name = n.name.split('.')[0]
                                    # Skip if it's a local module
                                    if module_name not in local_modules:
                                        imports.add(module_name)
                            elif isinstance(node, ast.ImportFrom) and node.module:
                                module_name = node.module.split('.')[0]
                                # Skip if it's a local module or relative import
                                if not node.level and module_name not in local_modules:
                                    imports.add(module_name)
                    except SyntaxError:
                        print(f"    Warning: Skipping {file_path} (syntax error)")
    
    # Also skip Python standard library modules
    if hasattr(sys, 'stdlib_module_names'):
        stdlib_modules = set(sys.stdlib_module_names)
        imports = imports - stdlib_modules
        print(f"  Excluded {len(stdlib_modules)} standard library modules")
    else:
        # Fallback for older Python versions - list common stdlib modules
        common_stdlib = {
            'os', 'sys', 're', 'json', 'time', 'datetime', 'math', 'random',
            'collections', 'itertools', 'functools', 'operator', 'typing',
            'pathlib', 'io', 'string', 'textwrap', 'copy', 'pickle',
            'subprocess', 'threading', 'multiprocessing', 'asyncio',
            'urllib', 'http', 'email', 'html', 'xml', 'csv', 'sqlite3',
            'logging', 'warnings', 'unittest', 'doctest', 'pdb',
            'argparse', 'configparser', 'hashlib', 'hmac', 'secrets',
            'uuid', 'socket', 'ssl', 'select', 'platform', 'locale',
            'codecs', 'encodings', 'base64', 'binascii', 'struct',
            'array', 'queue', 'heapq', 'bisect', 'weakref', 'types',
            'contextlib', 'abc', 'enum', 'dataclasses', 'importlib',
            'pkgutil', 'inspect', 'ast', 'dis', 'traceback', 'linecache',
            'shutil', 'tempfile', 'glob', 'fnmatch', 'stat', 'fileinput',
            'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile', 'zlib',
            'builtins', '__future__', 'gc', 'atexit', 'signal'
        }
        imports = imports - common_stdlib
        print(f"  Excluded {len(common_stdlib)} known standard library modules")
    
    print(f"  Scanned {file_count} Python files, found {len(imports)} third-party imports")
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

def fetch_local_package_license(package_name):
    """Fetch license from locally installed package using importlib.metadata."""
    try:
        # Try to get distribution metadata
        dist = importlib.metadata.distribution(package_name)
        
        # Try multiple metadata fields
        license_text = dist.metadata.get('License')
        if license_text and license_text.strip() and license_text.strip().upper() not in ['UNKNOWN', 'NONE']:
            # Try to determine license type from classifiers
            license_type = None
            classifiers = dist.metadata.get_all('Classifier') or []
            for classifier in classifiers:
                if classifier.startswith('License ::'):
                    parts = classifier.split(' :: ')
                    if len(parts) >= 3:
                        license_type = normalize_license_name(parts[2])
                        break
            
            if not license_type and license_text:
                # Try to infer from license text
                license_type = infer_license_from_text(license_text)
            
            return license_type, license_text
        
        # Try to find LICENSE file in package location
        if hasattr(dist, 'files') and dist.files:
            for file in dist.files:
                if file.name.upper() in [n.upper() for n in LICENSE_FILE_NAMES]:
                    try:
                        license_content = file.read_text()
                        if license_content:
                            license_type = infer_license_from_text(license_content)
                            return license_type, license_content
                    except Exception:
                        continue
                        
    except Exception:
        pass
    
    return None, None

def infer_license_from_text(text):
    """Infer license type from license text content."""
    if not text:
        return None
        
    text_lower = text.lower()
    
    # Common license detection patterns
    patterns = {
        'MIT': ['mit license', 'permission is hereby granted, free of charge'],
        'Apache-2.0': ['apache license', 'version 2.0', 'www.apache.org/licenses/'],
        'BSD-3-Clause': ['bsd 3-clause', 'redistribution and use in source and binary forms'],
        'BSD-2-Clause': ['bsd 2-clause', 'simplified bsd license'],
        'GPL-3.0': ['gnu general public license', 'version 3', 'gpl-3', 'gplv3'],
        'GPL-2.0': ['gnu general public license', 'version 2', 'gpl-2', 'gplv2'],
        'LGPL-3.0': ['gnu lesser general public license', 'lgpl', 'version 3'],
        'ISC': ['isc license', 'permission to use, copy, modify'],
        'MPL-2.0': ['mozilla public license', 'version 2.0', 'mpl-2.0'],
        'CC0-1.0': ['cc0', 'public domain', 'no copyright'],
        'Unlicense': ['unlicense', 'public domain', 'no conditions whatsoever'],
    }
    
    for license_type, keywords in patterns.items():
        if all(keyword in text_lower for keyword in keywords):
            return license_type
    
    # Fallback: check for common license URLs
    url_patterns = {
        'opensource.org/licenses/MIT': 'MIT',
        'opensource.org/licenses/Apache-2.0': 'Apache-2.0',
        'opensource.org/licenses/BSD-3-Clause': 'BSD-3-Clause',
        'gnu.org/licenses/gpl-3.0': 'GPL-3.0',
        'mozilla.org/MPL/2.0': 'MPL-2.0',
    }
    
    for url, license_type in url_patterns.items():
        if url in text:
            return license_type
    
    return None

def fetch_github_raw_license(repo):
    """Fetch license file directly from GitHub raw content."""
    try:
        base_url = f"https://raw.githubusercontent.com/{repo}"
        
        # Try main/master branch
        for branch in ['main', 'master']:
            for filename in LICENSE_FILE_NAMES[:8]:  # Try most common names
                url = f"{base_url}/{branch}/{filename}"
                try:
                    response = make_request_with_retry(url, timeout=3)
                    if response and response.ok:
                        license_text = response.text
                        license_type = infer_license_from_text(license_text)
                        return license_type, license_text
                except Exception:
                    continue
                    
    except Exception:
        pass
    
    return None, None

def fetch_pypi_license(package_name):
    """Fetch license information directly from PyPI."""
    try:
        response = make_request_with_retry(f"https://pypi.org/pypi/{package_name}/json")
        data = response.json()
        
        info = data['info']
        license_text = info.get('license', '') or ''  # Ensure it's never None
        
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
    # Load cache if available
    global license_cache
    cache_file = Path('.mdgen_cache.json')
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                license_cache = {k: (v['name'], v['text'], v['source']) for k, v in cache_data.items()}
                print(f"Loaded license cache with {len(license_cache)} entries")
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            license_cache = {}
    
    software_name, software_version, dev_name, dev_address = prompt_user()
    print("\nScanning source files and extracting imports...")
    third_party_modules = extract_imports_from_code(".")
    
    if not third_party_modules:
        print("\nNo third-party modules found. Only local modules detected.")
        modules_info = {}
        verification_warnings = []
    else:
        print(f"\nThird-party modules to check: {', '.join(third_party_modules)}")
        modules_info = {}
        verification_warnings = []
        print(f"\nFetching license information for {len(third_party_modules)} modules...")
        
        for i, module in enumerate(third_party_modules, 1):
            print(f"  [{i}/{len(third_party_modules)}] Checking {module}...")
            
            # Check cache first
            if module in license_cache:
                print(f"    Using cached license information")
                modules_info[module] = license_cache[module]
                continue
            
            # Try multiple sources in order of reliability
            license_sources = []
            
            # 1. Try local package metadata first (most reliable for installed packages)
            local_license_name, local_license_text = fetch_local_package_license(module)
            if local_license_name and local_license_text:
                license_sources.append(('local', local_license_name, local_license_text))
                print(f"    Found local package license: {local_license_name}")
            
            # 2. Try PyPI API
            pypi_license_name, pypi_license_text = fetch_pypi_license(module)
            if pypi_license_name and pypi_license_text:
                license_sources.append(('pypi', pypi_license_name, pypi_license_text))
                print(f"    Found PyPI license: {pypi_license_name}")
            
            # 3. Try GitHub if available
            repo = fetch_github_repo(module)
            if repo:
                print(f"    Found GitHub repo: {repo}")
                
                # Try GitHub API first
                github_license_name, github_license_text = fetch_github_license(repo)
                if github_license_name and github_license_text:
                    license_sources.append(('github_api', github_license_name, github_license_text))
                    print(f"    Found GitHub API license: {github_license_name}")
                
                # Try raw GitHub content as fallback
                if not github_license_name:
                    raw_license_name, raw_license_text = fetch_github_raw_license(repo)
                    if raw_license_name and raw_license_text:
                        license_sources.append(('github_raw', raw_license_name, raw_license_text))
                        print(f"    Found GitHub raw license: {raw_license_name}")
            
            # Validate and choose best license
            if license_sources:
                # Group by license type
                license_types = {}
                for source, name, text in license_sources:
                    normalized_name = normalize_license_name(name)
                    if normalized_name:
                        if normalized_name not in license_types:
                            license_types[normalized_name] = []
                        license_types[normalized_name].append((source, name, text))
                
                # If all sources agree, use the most complete text
                if len(license_types) == 1:
                    license_name = list(license_types.keys())[0]
                    # Choose the longest/most complete text
                    best_source = max(license_types[license_name], key=lambda x: len(x[2]))
                    source_type, _, license_text = best_source
                    
                    # Determine source description
                    sources = [s[0] for s in license_types[license_name]]
                    if len(sources) > 1:
                        source_desc = f"Multiple sources ({', '.join(sources)}) - verified"
                        print(f"    ✓ Verified license: {license_name} (consistent across {len(sources)} sources)")
                    else:
                        source_desc = f"{source_type} package"
                    
                    modules_info[module] = (license_name, license_text, source_desc)
                    license_cache[module] = (license_name, license_text, source_desc)
                else:
                    # Sources disagree - log warning and use most reliable source
                    license_list = list(license_types.keys())
                    warning = f"License mismatch for {module}: {', '.join([f'{k}({len(v)} sources)' for k, v in license_types.items()])}"
                    verification_warnings.append(warning)
                    print(f"    ⚠ License mismatch: {', '.join(license_list)}")
                    
                    # Priority order: local > pypi > github_api > github_raw
                    for source_priority in ['local', 'pypi', 'github_api', 'github_raw']:
                        for license_name, sources in license_types.items():
                            for source, _, text in sources:
                                if source == source_priority:
                                    modules_info[module] = (license_name, text, f"{source} (conflict resolved)")
                                    license_cache[module] = (license_name, text, f"{source} (conflict resolved)")
                                    print(f"    Using {source} license: {license_name}")
                                    break
                            if module in modules_info:
                                break
                        if module in modules_info:
                            break
            else:
                print(f"    No license information available from any source")
    
    # Report verification warnings
    if verification_warnings:
        print(f"\n⚠ License verification warnings:")
        for warning in verification_warnings:
            print(f"  - {warning}")
    
    print(f"\n{len(modules_info)} third-party components found with known licenses.")
    
    # Save cache for future runs
    cache_file = Path('.mdgen_cache.json')
    try:
        with open(cache_file, 'w') as f:
            # Convert cache to JSON-serializable format
            cache_data = {k: {'name': v[0], 'text': v[1], 'source': v[2]} for k, v in license_cache.items()}
            json.dump(cache_data, f, indent=2)
        print(f"License cache saved to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")
    
    eula_text = build_eula(software_name, software_version, dev_name, dev_address, modules_info)

    with open("LICENSE", "w", encoding="utf-8") as f:
        f.write(eula_text)
    print("\nLicense information generated and saved as LICENSE")

if __name__ == "__main__":
    main()
