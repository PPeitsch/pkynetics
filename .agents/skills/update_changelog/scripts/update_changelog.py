import argparse
import datetime
import re
import sys

def main():
    parser = argparse.ArgumentParser(description="Update CHANGELOG.md with a new release entry securely.")
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

    # Basic version format validation
    if not re.match(r'^v\d+\.\d+\.\d+$', args.version):
        print("Error: Version must follow the 'vX.Y.Z' format (e.g. v1.0.0)")
        sys.exit(1)

    # Replace literal \n with actual newlines from shell args to prevent mangling strings
    def format_section(title, text):
        if not text:
            return ""
        # decode escaped newlines that might come through bash
        text = text.replace('\\n', '\n')
        # ensure it ends with single newline
        text = text.strip() + '\n'
        return f"### {title}\n{text}\n"

    entry = f"## [{args.version}] - {args.date}\n\n"
    entry += format_section("Added", args.added)
    entry += format_section("Changed", args.changed)
    entry += format_section("Fixed", args.fixed)
    entry += format_section("Security", args.security)

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
