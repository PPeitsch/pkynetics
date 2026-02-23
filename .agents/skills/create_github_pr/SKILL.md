---
name: create_github_pr
description: Create GitHub Pull Requests automatically.
---

# Create GitHub PR Skill

This skill allows the agent to automatically create new GitHub Pull Requests directly from the terminal.

## Capabilities

1. **Create PR**: Open a new Pull Request against any branch in the repository.

## Scripts

- **Script Path**: `.agents/skills/create_github_pr/scripts/create_pr.py`
- **Dependencies**: Requires GitHub CLI (`gh`) to be installed and authenticated (`gh auth login`). Note: Make sure to commit and push your branch to the remote repository (`git push -u origin <branch_name>`) before using this tool.

## Usage

### Create a new PR
Before creating the PR, push your local commits to the target remote branch. Next, run the script:

```bash
python .agents/skills/create_github_pr/scripts/create_pr.py --title "Feature: Add amazing thing" --body "This PR fixes..." --base main
```

*Arguments:*
- `--title` (Required): The PR title.
- `--body` (Required): Description of the PR changes.
- `--base` (Optional): The branch you want your changes pulled into (default: `main`).
- `--draft` (Optional): Pass this flag without value to create a draft PR.
- `--labels` (Optional): Comma-separated labels.
