import argparse
import datetime
import re
import sys

def main():
    parser = argparse.ArgumentParser(description="Update CHANGELOG.md with a new release entry.")
    parser.add_argument("--version", required=True, help="Release version (e.g. v1.0.0)")
    parser.add_argument("--date", default=datetime.datetime.now().strftime("%Y-%m-%d"), help="Release date (YYYY-MM-DD)")
    parser.add_argument("--added", help="Content for 'Added' section")
    parser.add_argument("--changed", help="Content for 'Changed' section")
    parser.add_argument("--fixed", help="Content for 'Fixed' section")
    parser.add_argument("--security", help="Content for 'Security' section")
    args = parser.parse_args()

    changelog_path = "CHANGELOG.md"
    try:
        with open(changelog_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading CHANGELOG.md: {e}")
        sys.exit(1)

    entry = f"## [{args.version}] - {args.date}\n\n"
    
    if args.added:
        entry += f"### Added\n{args.added.strip()}\n\n"
    if args.changed:
        entry += f"### Changed\n{args.changed.strip()}\n\n"
    if args.fixed:
        entry += f"### Fixed\n{args.fixed.strip()}\n\n"
    if args.security:
        entry += f"### Security\n{args.security.strip()}\n\n"

    if entry.strip() == f"## [{args.version}] - {args.date}":
        print("Error: No changes provided. Please provide at least one section: --added, --changed, --fixed, or --security.")
        sys.exit(1)

    # Find the position of the first version header to insert just above it
    match = re.search(r'## \[v', content)
    if not match:
        insert_idx = len(content)
    else:
        insert_idx = match.start()

    new_content = content[:insert_idx] + entry + "\n" + content[insert_idx:]

    try:
        with open(changelog_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Successfully added version {args.version} to CHANGELOG.md")
    except Exception as e:
        print(f"Error writing to {changelog_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
