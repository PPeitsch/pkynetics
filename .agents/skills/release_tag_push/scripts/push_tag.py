import subprocess
import argparse
import sys
import time
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def run_cmd(args):
    result = subprocess.run(args, capture_output=True, encoding='utf-8', check=False)
    if result.returncode != 0:
        print(f"Error executing ({' '.join(args)}): {result.stderr}")
        return False, result.stderr
    return True, result.stdout.strip()

def main():
    parser = argparse.ArgumentParser(description="Wait for main branch CI workflows to finish, then push tag.")
    parser.add_argument("version", help="Version tag to push, e.g. v1.0.0")
    args = parser.parse_args()

    # Step 1: Wait for workflows on main
    print("Checking for running workflows on 'main' branch...")
    # Sleep to allow GitHub actions to detect the recent push trigger
    time.sleep(5)
    
    success, output = run_cmd(['gh', 'run', 'list', '--branch', 'main', '--limit', '1', '--json', 'databaseId,status,conclusion'])
    if not success:
        print("Failed to poll GitHub runs. Please check your GitHub CLI authentication.")
        sys.exit(1)
        
    runs = json.loads(output)
    if runs:
        run = runs[0]
        if run['status'] in ['in_progress', 'queued']:
            print(f"Waiting for CI workflow run {run['databaseId']} to complete on main...")
            # Watch command streams live status until finished
            subprocess.run(['gh', 'run', 'watch', str(run['databaseId'])], check=False)
            
            # Re-check conclusion state
            success, view_output = run_cmd(['gh', 'run', 'view', str(run['databaseId']), '--json', 'conclusion'])
            if success:
                run_view = json.loads(view_output)
                if run_view.get('conclusion') != 'success':
                    print("Error: The main branch workflow failed! Aborting tag release.")
                    sys.exit(1)
        elif run['conclusion'] != 'success':
            print("Error: The latest workflow on main failed or was cancelled. Aborting tag release.")
            sys.exit(1)
        else:
            print("Latest workflow on main succeeded.")
    else:
        print("No prior workflows found on main branch. Assuming safe to proceed.")
    
    # Step 2: Push tag
    print(f"Creating local tag {args.version} (if it doesn't exist already)...")
    run_cmd(['git', 'tag', args.version])
    
    print(f"Pushing tag {args.version} specifically to origin...")
    success, output = run_cmd(['git', 'push', 'origin', args.version])
    
    if success:
        print("Tag pushed successfully!")
    else:
        # Check if already exists error
        if "already exists" in output.lower() or "rejected" in output.lower():
             print(f"Notice: Tag pushing failed. It might already exist on remote. Details: {output}")
        else:
             print("Critical error trying to push tag. Aborting.")
             sys.exit(1)

if __name__ == "__main__":
    main()
