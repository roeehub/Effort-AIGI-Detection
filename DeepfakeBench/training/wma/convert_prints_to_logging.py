#!/usr/bin/env python3
"""
Script to convert print() statements to logging calls in server.py
"""

import re

def convert_prints_to_logging(file_path):
    """Convert print statements to appropriate logging calls."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Convert different types of print statements to appropriate logging levels
    
    # Warnings and errors
    content = re.sub(r'print\(f"\[DEBUG\] ⚠️', r'logging.warning(f"[DEBUG] ⚠️', content)
    content = re.sub(r'print\(f"⚠️', r'logging.warning(f"⚠️', content)
    content = re.sub(r'print\(f"\[.*?\] Error', r'logging.error(f"[', content)
    content = re.sub(r'print\(f"\[.*?\] Failed', r'logging.error(f"[', content)
    content = re.sub(r'print\("\[.*?\] Error', r'logging.error("[', content)
    
    # Regular print statements
    content = re.sub(r'\bprint\(f"', r'logging.info(f"', content)
    content = re.sub(r'\bprint\(f\'', r'logging.info(f\'', content)
    content = re.sub(r'\bprint\("', r'logging.info("', content)
    content = re.sub(r'\bprint\(\'', r'logging.info(\'', content)
    content = re.sub(r'\bprint\(\n', r'logging.info(\n', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Successfully converted print statements to logging in {file_path}")

if __name__ == '__main__':
    convert_prints_to_logging('server.py')
