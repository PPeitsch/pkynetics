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
    parser = argparse.ArgumentParser(description="Create a new GitHub Issue.")
    parser.add_argument("--title", type=str, required=True, help="Title of the issue.")
    parser.add_argument("--body", type=str, required=True, help="Body text of the issue.")
    parser.add_argument("--labels", type=str, help="Comma-separated labels.")
    args = parser.parse_args()

    command = ['issue', 'create', '--title', args.title, '--body', args.body]
    if args.labels:
        command.extend(['--label', args.labels])
    
    print("Creating GitHub Issue...")
    output = run_gh_command(command)
    print("Issue created successfully!")
    print(output.strip())

if __name__ == "__main__":
    main()
