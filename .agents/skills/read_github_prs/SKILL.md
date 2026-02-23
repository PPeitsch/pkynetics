---
name: read_github_prs
description: Read and analyze GitHub Pull Requests systematically.
---

# GitHub PR Reader Skill

This skill allows the agent to automatically read and analyze GitHub Pull Requests in the repository.

## Capabilities

1. **List PRs**: Output a summarized markdown list of pull requests using the helper script.
2. **Read Specific PR**: Use GitHub CLI (`gh`) to deeply read PR content, comments, and diffs.

## Scripts

- **Script Path**: `.agents/skills/read_github_prs/scripts/read_prs.py`
- **Dependencies**: Requires GitHub CLI (`gh`) to be installed and authenticated (`gh auth login`).

## Usage

### 1. View Summary of Pull Requests
Run the helper script to list PRs clearly:
```bash
python .agents/skills/read_github_prs/scripts/read_prs.py --limit 10 --state open
```

### 2. View a Specific Pull Request
To review a pull request, read the body, comments, and the code diff:
```bash
gh pr view <PR_NUMBER> --comments
gh pr diff <PR_NUMBER>  # View code changes
```

### 3. Search PRs
Search specifically using a query parameter:
```bash
gh pr list --search "fix" --state all
```
