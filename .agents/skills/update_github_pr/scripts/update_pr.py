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
    parser = argparse.ArgumentParser(description="Update an existing GitHub Pull Request.")
    parser.add_argument("pr_number", type=str, help="The PR number to update.")
    parser.add_argument("--title", type=str, help="New title of the pull request.")
    parser.add_argument("--body", type=str, help="New body text of the pull request.")
    parser.add_argument("--base", type=str, help="Change the base branch.")
    parser.add_argument("--add-label", type=str, help="Comma-separated labels to add.")
    parser.add_argument("--remove-label", type=str, help="Comma-separated labels to remove.")
    args = parser.parse_args()

    command = ['pr', 'edit', args.pr_number]
    
    if args.title:
        command.extend(['--title', args.title])
    if args.body:
        command.extend(['--body', args.body])
    if args.base:
        command.extend(['--base', args.base])
    if args.add_label:
        command.extend(['--add-label', args.add_label])
    if args.remove_label:
        command.extend(['--remove-label', args.remove_label])
        
    if len(command) == 3:
        print("Error: No modifications requested.")
        sys.exit(1)
    
    print(f"Updating GitHub Pull Request #{args.pr_number}...")
    output = run_gh_command(command)
    print(f"Pull Request #{args.pr_number} updated successfully!")
    print(output.strip())

if __name__ == "__main__":
    main()
