import os
import re
import ast
import requests
from datetime import datetime
from pathlib import Path

# Constants
EULA_TEMPLATE_URL = "https://www.apple.com/legal/sla/docs/macOSSequoia.pdf"
MIT_LICENSE_URL = "https://raw.githubusercontent.com/github/choosealicense.com/gh-pages/_licenses/mit.txt"
GITHUB_API_BASE = "https://api.github.com/repos"

def prompt_user():
    print("Please enter the following details for the EULA:")
    software_name = input("Software Name: ").strip()
    developer_name = input("Developer Name: ").strip()
    developer_address = input("Developer Address: ").strip()
    return software_name, developer_name, developer_address

def extract_imports_from_code(base_dir):
    imports = set()
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for n in node.names:
                                    imports.add(n.name.split('.')[0])
                            elif isinstance(node, ast.ImportFrom) and node.module:
                                imports.add(node.module.split('.')[0])
                    except SyntaxError:
                        pass  # Skip files with invalid syntax
    return sorted(imports)

def fetch_github_repo(package_name):
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=5)
        response.raise_for_status()
        data = response.json()
        repo_url = data['info']['project_urls'].get('Source', '') or data['info'].get('home_page', '')
        if "github.com" in repo_url:
            parts = repo_url.split("github.com/")[-1].split("/")
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

def build_eula(software_name, developer_name, developer_address, modules_info):
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
    return eula

def main():
    software_name, dev_name, dev_address = prompt_user()
    print("\nScanning source files and extracting imports...")
    third_party_modules = extract_imports_from_code(".")

    modules_info = {}
    for module in third_party_modules:
        repo = fetch_github_repo(module)
        if repo:
            license_name, license_text = fetch_github_license(repo)
            if license_name and license_text:
                modules_info[module] = (license_name, license_text, repo)
    
    print(f"\n{len(modules_info)} third-party components found with known licenses.")
    eula_text = build_eula(software_name, dev_name, dev_address, modules_info)

    with open("LICENSE.md", "w", encoding="utf-8") as f:
        f.write(eula_text)
    print("\nLicense information generated and saved as LICENSE.md")

if __name__ == "__main__":
    main()
