---
name: read_github_issues
description: Read and analyze GitHub Issues systematically.
---

# GitHub Issues Reader Skill

This skill allows the agent to automatically read and analyze GitHub Issues in the repository.

## Capabilities

1. **List Issues**: Output a summarized markdown list of issues using the provided helper script.
2. **Read Specific Issue**: Use the GitHub CLI (`gh`) to deeply read content and comments.

## Scripts

- **Script Path**: `.agents/skills/read_github_issues/scripts/read_issues.py`
- **Dependencies**: Requires GitHub CLI (`gh`) to be installed and authenticated (`gh auth login`).

## Usage

### 1. View Summary of Issues
Run the helper script to list issues clearly:
```bash
python .agents/skills/read_github_issues/scripts/read_issues.py --limit 10 --state open
```

### 2. View a Specific Issue
Once you find an issue number, read its detailed body and comments using `gh`:
```bash
gh issue view <ISSUE_NUMBER> --comments
```

### 3. Search and filter Issues
Search specifically using the query parameter to locate bugs or keywords:
```bash
gh issue list --search "bug" --state all
```
