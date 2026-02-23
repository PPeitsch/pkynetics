---
name: release_tag_push
description: Safely wait for main branch workflows to complete before generating and pushing a version tag.
---

# Release Tag Push Skill

This skill allows the agent to safely execute the final steps of a deployment/release. When a release commit is pushed to the main branch, a test-and-publish workflow runs. Pushing the tag *at the same time* causes collision and redundant or failed workflow executions. This script solves that by dynamically waiting on the GitHub API for the primary workflows to succeed before dispatching the tag.

## Capabilities

1. **Wait for CI Validation**: Checks `gh run list` and `gh run watch` to ensure the branch passed tests.
2. **Tag & Push**: Generates the `git tag` locally and specifically pushes only that isolated tag to the remote.

## Scripts

- **Script Path**: `.agents/skills/release_tag_push/scripts/push_tag.py`
- **Dependencies**: Requires GitHub CLI (`gh`) to be installed and authenticated (`gh auth login`).

## Usage

### Execute safe tag push
Use this specifically after a standard code commit push (e.g. `git push origin main`) to conclude a release process.

```bash
python .agents/skills/release_tag_push/scripts/push_tag.py v0.4.6
```

*Arguments:*
- `version` (Required): The tag string you intend to release, e.g., `v1.2.3` (positional argument).
