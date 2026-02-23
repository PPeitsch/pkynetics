import subprocess
import json
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def run_gh_command(args):
    try:
        # Using check=True to raise an exception on non-zero exit code
        result = subprocess.run(['gh'] + args, capture_output=True, encoding='utf-8', check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing gh command: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: GitHub CLI (gh) is not installed or not in PATH.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Read Pull Requests from GitHub for the agent.")
    parser.add_argument("--limit", type=int, default=10, help="Number of PRs to fetch.")
    parser.add_argument("--state", type=str, default="open", choices=["open", "closed", "all"], help="State of the PRs.")
    args = parser.parse_args()

    print(f"# GitHub Pull Requests (Limit: {args.limit}, State: {args.state})\n")
    
    prs_json = run_gh_command([
        'pr', 'list', 
        '--state', args.state, 
        '--limit', str(args.limit), 
        '--json', 'number,title,state,author,createdAt,isDraft'
    ])
    
    prs = json.loads(prs_json)
    if not prs:
        print("No pull requests found.")
        return
        
    for pr in prs:
        author = pr.get('author', {}).get('login', 'Unknown')
        draft_status = " (Draft)" if pr.get('isDraft') else ""
        print(f"- **#{pr['number']}** [{pr['state'].upper()}]{draft_status} {pr['title']} (by {author})")

if __name__ == "__main__":
    main()
