#!/usr/bin/env python3
"""
Validate Jupyter Book structure before building.
Checks for missing files, broken links, and configuration issues.
"""

import os
import sys
import yaml
from pathlib import Path

def check_file_exists(filepath):
    """Check if file exists (with or without extension)."""
    base = Path(filepath)
    if base.exists():
        return True
    if (base.parent / f"{base.name}.md").exists():
        return True
    if (base.parent / f"{base.name}.ipynb").exists():
        return True
    return False

def validate_toc():
    """Validate table of contents."""
    print("Validating _toc.yml...")
    issues = []
    
    toc_path = Path("_toc.yml")
    if not toc_path.exists():
        issues.append("_toc.yml not found")
        return issues
    
    with open(toc_path, 'r') as f:
        try:
            toc = yaml.safe_load(f)
        except yaml.YAMLError as e:
            issues.append(f"Invalid YAML in _toc.yml: {e}")
            return issues
    
    def check_entry(entry, path=""):
        """Recursively check TOC entries."""
        local_issues = []
        if isinstance(entry, dict):
            if 'file' in entry:
                filepath = entry['file']
                full_path = os.path.join(path, filepath) if path else filepath
                if not check_file_exists(full_path):
                    local_issues.append(f"Missing file: {full_path}")
            if 'chapters' in entry:
                for chapter in entry['chapters']:
                    local_issues.extend(check_entry(chapter, path))
        elif isinstance(entry, list):
            for item in entry:
                local_issues.extend(check_entry(item, path))
        return local_issues
    
    issues.extend(check_entry(toc.get('chapters', [])))
    
    if issues:
        print("  ✗ Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✓ All TOC references are valid")
    
    return issues

def validate_config():
    """Validate _config.yml."""
    print("\nValidating _config.yml...")
    issues = []
    
    config_path = Path("_config.yml")
    if not config_path.exists():
        issues.append("_config.yml not found")
        return issues
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            issues.append(f"Invalid YAML in _config.yml: {e}")
            return issues
    
    required_fields = ['title']
    for field in required_fields:
        if field not in config:
            issues.append(f"Missing required field: {field}")
    
    if issues:
        print("  ✗ Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✓ Configuration is valid")
    
    return issues

def validate_structure():
    """Validate directory structure."""
    print("\nValidating directory structure...")
    issues = []
    
    required_dirs = [
        'tutorials',
        'concepts',
        'guides',
        'api',
        'use_cases'
    ]
    
    for dirname in required_dirs:
        dirpath = Path(dirname)
        if not dirpath.exists():
            issues.append(f"Missing directory: {dirname}")
        elif not dirpath.is_dir():
            issues.append(f"Not a directory: {dirname}")
    
    if issues:
        print("  ✗ Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✓ Directory structure is valid")
    
    return issues

def main():
    """Run all validations."""
    print("=" * 60)
    print("Jupyter Book Structure Validation")
    print("=" * 60)
    
    all_issues = []
    all_issues.extend(validate_toc())
    all_issues.extend(validate_config())
    all_issues.extend(validate_structure())
    
    print("\n" + "=" * 60)
    if all_issues:
        print(f"✗ Validation failed: {len(all_issues)} issue(s) found")
        sys.exit(1)
    else:
        print("✓ All validations passed!")
        print("\nReady for Jupyter Book build.")
        sys.exit(0)

if __name__ == '__main__':
    main()

