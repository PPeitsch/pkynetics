import subprocess
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def run_gh_command(args):
    try:
        result = subprocess.run(['gh'] + args, capture_output=True, encoding='utf-8', check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing gh command: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: GitHub CLI (gh) is not installed or not in PATH.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Create a new GitHub Pull Request.")
    parser.add_argument("--title", type=str, required=True, help="Title of the pull request.")
    parser.add_argument("--body", type=str, required=True, help="Body text of the pull request.")
    parser.add_argument("--base", type=str, default="main", help="Base branch (default: main).")
    parser.add_argument("--draft", action="store_true", help="Create as a draft PR.")
    parser.add_argument("--labels", type=str, help="Comma-separated labels.")
    args = parser.parse_args()

    command = ['pr', 'create', '--title', args.title, '--body', args.body, '--base', args.base]
    if args.draft:
        command.append('--draft')
    if args.labels:
        command.extend(['--label', args.labels])
    
    print("Creating GitHub Pull Request...")
    output = run_gh_command(command)
    print("Pull Request created successfully!")
    print(output.strip())

if __name__ == "__main__":
    main()
