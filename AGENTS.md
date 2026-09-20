# AGENTS.md — Pkynetics

> **CRITICAL RULE**: never run `git commit` or `git push` without first running the full
> quality pipeline — `black`, `isort`, `mypy`, `pytest` — successfully.

Python library for thermal analysis kinetic methods: data import from thermal analysis
instruments (TGA, DSC, dilatometry), model-free and model-fitting kinetic analysis, and
result visualization.

## Environment

Developed on **Linux (Ubuntu)**, also used on **Windows** — detect the host and adapt
(`source .venv/bin/activate` vs `.venv\Scripts\Activate.ps1`). Ask if unsure.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

## The quality pipeline

Run all four before every commit. This is what CI runs.

```bash
black --check . && isort --check-only . && mypy src && pytest --cov
```

`black` and `isort` apply with `black .` and `isort .` (line length 88, isort profile
`black`). `mypy` is in **strict** mode. CI additionally builds the sdist and runs the
tests against the installed wheel, on Python 3.10–3.13.

## Conventions

- **Commits** — Conventional Commits: `type: description`, with
  `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `release`.
- **Type hints** on every new function and method; **NumPy-style docstrings**.
- **Tests** for new functionality.
- **`CHANGELOG.md`** updated for any user-facing change
  ([Keep a Changelog](https://keepachangelog.com/en/1.0.0/)).
- **88-character lines** (black default).

## Release

1. Tests green locally.
2. Bump `src/pkynetics/__about__.py` **and** `docs/conf.py` — the version is not yet
   centralised, so they drift unless updated together.
3. `CHANGELOG.md` with the release date.
4. `release: version X.Y.Z`, then `git tag vX.Y.Z && git push origin vX.Y.Z`.
5. CI publishes to PyPI from the distributions the Package job tested.

## Skills submodule

Workflows live in the `skills/` submodule, which is a **private** repository
and is not needed to build, test or use the library — only to cut a release
and to drive GitHub. A plain `git clone` is the right way in; `--recursive`
just fails for anyone without access. **Use the wrapper, not the individual
scripts**, and prefer these over raw shell commands for GitHub and release work:

```bash
python skills/run_skill.py read_github_issues --limit <N>
python skills/run_skill.py read_github_prs --limit <N>
python skills/run_skill.py create_github_issue --title "..." --body "..."
python skills/run_skill.py create_github_pr --title "..." --body "..."
python skills/run_skill.py update_github_issue <issue_number> ...
python skills/run_skill.py update_github_pr <pr_number> ...
python skills/run_skill.py update_changelog ...
python skills/run_skill.py release_tag_push ...
```

Full list in `skills/tools/`. Docs are Sphinx (`docs/`, Napoleon, NumPy docstrings),
published to ReadTheDocs.
