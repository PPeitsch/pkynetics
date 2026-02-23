---
name: update_changelog
description: Safely update the project's CHANGELOG.md with a new version entry.
---

# Update Changelog Skill

This skill allows the agent to safely parse and prepend a new version release inside `CHANGELOG.md`, adhering to the "Keep a Changelog" formatting standards without destructive overrides.

## Capabilities

1. **Update CHANGELOG**: Prepend a new version block (`## [vX.Y.Z] - Date`) keeping existing history intact.

## Scripts

- **Script Path**: `.agents/skills/update_changelog/scripts/update_changelog.py`

## Usage

### Add a new release
Use this script specifying the version and whichever sections you have changes for (`--added`, `--changed`, `--fixed`, `--security`). Be sure to use markdown bullet points for the content strings `"- Item 1\n- Item 2"`.

```bash
python .agents/skills/update_changelog/scripts/update_changelog.py --version "v0.4.6" --added "- New feature A\n- New feature B" --fixed "- Bug fixed"
```

*Arguments:*
- `--version` (Required): The tag string, e.g., `v1.2.3`.
- `--date` (Optional): Override current date using format `YYYY-MM-DD`.
- `--added` (Optional): String block of added features.
- `--changed` (Optional): String block of changes.
- `--fixed` (Optional): String block of bug fixes.
- `--security` (Optional): String block of security fixes.
