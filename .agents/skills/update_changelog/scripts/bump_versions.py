import argparse
import sys
import re
import os

def update_file(filepath, pattern, replacement, success_msg):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content, count = re.subn(pattern, replacement, content, count=1)
        
        if count == 0:
            print(f"Warning: Could not find version pattern in {filepath}")
            return False
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        print(success_msg)
        return True
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Recursively bump project versions across configuration files.")
    parser.add_argument("version", help="New version to set (e.g. 0.4.7 or v0.4.7)")
    args = parser.parse_args()

    # Normalize version string to remove leading 'v' if passed by accident
    clean_version = args.version.lstrip('v')
    print(f"Bumping project version to: {clean_version}")

    success = True

    # 1. Update src/pkynetics/__about__.py
    about_path = os.path.join("src", "pkynetics", "__about__.py")
    success &= update_file(
        about_path,
        r'__version__\s*=\s*([\'"]).*?\1',
        f'__version__ = "{clean_version}"',
        f"✓ Updated __about__.py to {clean_version}"
    )

    # 2. Update docs/conf.py
    docs_path = os.path.join("docs", "conf.py")
    success &= update_file(
        docs_path,
        r'release\s*=\s*([\'"]).*?\1',
        f'release = "{clean_version}"',
        f"✓ Updated docs/conf.py to {clean_version}"
    )

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
