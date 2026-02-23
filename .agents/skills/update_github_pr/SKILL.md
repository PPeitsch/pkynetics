---
name: update_github_pr
description: Edit and modify existing GitHub Pull Requests automatically.
---

# Update GitHub PR Skill

This skill allows the agent to dynamically edit the properties (title, body, labels, base branch) of an already created GitHub Pull Request.

## Capabilities

1. **Update PR**: Modify an open or draft Pull Request.

## Scripts

- **Script Path**: `.agents/skills/update_github_pr/scripts/update_pr.py`
- **Dependencies**: Requires GitHub CLI (`gh`) to be installed and authenticated (`gh auth login`).

## Usage

### Edit an existing PR
To alter an existing PR without re-opening a new one, you can run:

```bash
python .agents/skills/update_github_pr/scripts/update_pr.py 123 --title "Fix: Re-worded correct title" --base main
```

*Arguments:*
- `pr_number` (Required): The numerical ID of the pull request (positional argument).
- `--title` (Optional): The new title.
- `--body` (Optional): Replacing description.
- `--base` (Optional): Change the base branch (e.g. `main` or `develop`).
- `--add-label` (Optional): Comma-separated list of labels to append.
- `--remove-label` (Optional): Comma-separated list of labels to detach.
