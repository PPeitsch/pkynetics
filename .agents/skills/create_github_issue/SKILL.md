---
name: create_github_issue
description: Create GitHub Issues automatically.
---

# Create GitHub Issue Skill

This skill allows the agent to automatically create new GitHub Issues in the repository.

## Capabilities

1. **Create Issue**: Create a new issue with a specific title, body, and optional labels.

## Scripts

- **Script Path**: `.agents/skills/create_github_issue/scripts/create_issue.py`
- **Dependencies**: Requires GitHub CLI (`gh`) to be installed and authenticated (`gh auth login`).

## Usage

### Create a new issue
Run the creation script providing the required title and body. Use quotes around your strings.

```bash
python .agents/skills/create_github_issue/scripts/create_issue.py --title "Bug: Description" --body "Steps to reproduce..." --labels "bug,help wanted"
```

*Arguments:*
- `--title` (Required): The issue title.
- `--body` (Required): The full issue text markdown. 
- `--labels` (Optional): A comma-separated list of labels to assign to the issue.
