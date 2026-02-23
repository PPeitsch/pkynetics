---
name: update_github_issue
description: Edit and modify existing GitHub Issues automatically.
---

# Update GitHub Issue Skill

This skill allows the agent to edit the title, body, or labels of existing GitHub Issues in the repository.

## Capabilities

1. **Update Issue**: Modify an existing issue's title, body, or labels.

## Scripts

- **Script Path**: `.agents/skills/update_github_issue/scripts/update_issue.py`
- **Dependencies**: Requires GitHub CLI (`gh`) to be installed and authenticated (`gh auth login`).

## Usage

### Edit an existing issue
Run the update script providing the issue number and the specific fields you want to update.

```bash
python .agents/skills/update_github_issue/scripts/update_issue.py 123 --title "Updated Issue Title" --add-label "bug,high-priority" --remove-label "needs-triage"
```

*Arguments:*
- `issue_number` (Required): The ID/number of the issue to update (positional argument).
- `--title` (Optional): The new issue title.
- `--body` (Optional): The new issue format/text.
- `--add-label` (Optional): Comma-separated labels to add.
- `--remove-label` (Optional): Comma-separated labels to remove.
