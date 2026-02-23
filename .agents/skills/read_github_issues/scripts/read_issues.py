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
    parser = argparse.ArgumentParser(description="Read Issues from GitHub for the agent.")
    parser.add_argument("--limit", type=int, default=10, help="Number of issues to fetch.")
    parser.add_argument("--state", type=str, default="open", choices=["open", "closed", "all"], help="State of the issues.")
    args = parser.parse_args()

    print(f"# GitHub Issues (Limit: {args.limit}, State: {args.state})\n")
    
    issues_json = run_gh_command([
        'issue', 'list', 
        '--state', args.state, 
        '--limit', str(args.limit), 
        '--json', 'number,title,state,author,createdAt,labels'
    ])
    
    issues = json.loads(issues_json)
    if not issues:
        print("No issues found.")
        return
        
    for issue in issues:
        labels = ", ".join([label['name'] for label in issue.get('labels', [])])
        author = issue.get('author', {}).get('login', 'Unknown')
        print(f"- **#{issue['number']}** [{issue['state'].upper()}] {issue['title']} (by {author}) - Labels: *{labels}*")

if __name__ == "__main__":
    main()
