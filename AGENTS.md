# Pkynetics Development Protocol for AI Agents

## OS Context

This project is developed primarily on **Linux (Ubuntu)** but may also be used on **Windows**. Detect the host OS and adapt commands accordingly:

- **Linux**: Use `source .venv/bin/activate`, paths with `/`
- **Windows**: Use `.venv\Scripts\Activate.ps1`, paths with `\`

When in doubt, ask the user which environment they are working on.

---

## Project Overview

**Pkynetics** is a Python library for thermal analysis kinetic methods, including:
- Data import from various thermal analysis instruments (TGA, DSC, Dilatometry)
- Model-free and model-fitting kinetic analysis methods
- Result visualization and plotting utilities

## Build System & Tooling

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **hatch** | Build system | `pyproject.toml` |
| **black** | Code formatting | `line-length = 88`, `target-version = ["py39"]` |
| **isort** | Import sorting | `profile = "black"`, `line_length = 88` |
| **mypy** | Type checking | Strict mode, see `pyproject.toml` |
| **pytest** | Testing | With `pytest-cov` for coverage |
| **sphinx** | Documentation | Published to ReadTheDocs |

## Python Version Support

- Python 3.9, 3.10, 3.11

## Development Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install in development mode with dev dependencies
pip install -e .[dev]
```

## Code Quality Commands

### Formatting
```bash
# Check formatting
black --check src/pkynetics tests

# Apply formatting
black src/pkynetics tests

# Check import sorting
isort --check-only src/pkynetics tests

# Apply import sorting
isort src/pkynetics tests
```

### Type Checking
```bash
mypy
```

### Testing
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src/pkynetics --cov-report=term-missing

# Run specific test file
pytest tests/test_specific.py -v
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/test-and-publish.yaml`) runs:

1. **Code Quality** (Python 3.11):
   - black --check
   - isort --check-only
   - mypy

2. **Tests** (Python 3.9, 3.10, 3.11):
   - pytest with coverage

3. **Publish** (on tag push `v*`):
   - Build and publish to PyPI

## Commit Message Format

Follow the Conventional Commits format:
```
type: description

- Detail 1
- Detail 2
```

Types: `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`, `test:`, `release:`

## Version Management

**Current state**: Version is defined in `src/pkynetics/__about__.py`.

> **Note**: Version centralization is planned for future improvement. Currently `docs/conf.py` must be updated manually to match.

### Updating Version
1. Update `src/pkynetics/__about__.py`
2. Update `docs/conf.py` (manually, for now)
3. Update `CHANGELOG.md`

## Release Process

1. Ensure all tests pass locally
2. Update version in `__about__.py` and `docs/conf.py`
3. Update `CHANGELOG.md` with release date
4. Commit: `release: version X.Y.Z`
5. Create and push tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
6. CI will automatically publish to PyPI

## CHANGELOG Format

Follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/):

```markdown
## [vX.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing functionality

### Fixed
- Bug fixes

### Security
- Security patches
```

## Documentation

- Source: `docs/` directory
- Build: Sphinx with Napoleon extension
- Hosted: ReadTheDocs
- Docstring format: NumPy style

## Project Structure

```
pkynetics/
├── src/pkynetics/           # Main package
│   ├── __about__.py         # Version info
│   ├── data_import/         # Data importers
│   ├── data_preprocessing/  # Preprocessing utilities
│   ├── model_fitting_methods/
│   ├── model_free_methods/
│   ├── result_visualization/
│   ├── synthetic_data/
│   └── technique_analysis/
├── tests/                   # Test suite
├── docs/                    # Sphinx documentation
├── examples/                # Usage examples
├── .github/                 # GitHub Actions, templates
├── pyproject.toml          # Project configuration
├── CHANGELOG.md            # Version history
└── README.md               # Project overview
```

## Key Guidelines for AI Agents

1. **Always run quality checks** before committing: `black`, `isort`, `mypy`, `pytest`
2. **Use type hints** for all new functions and methods
3. **Write tests** for new functionality
4. **Update CHANGELOG.md** for any user-facing changes
5. **Follow NumPy docstring format** for documentation
6. **Respect the 88-character line limit** (black default)

## Available AI Agent Skills

This repository has specific skills designed to help AI agents navigate project context. Before working on new features or checking for issues, agents should use the following skills located in `.agents/skills/`:

- **read_github_issues**: Read open/closed issues using `python .agents/skills/read_github_issues/scripts/read_issues.py --limit <N>`.
- **read_github_prs**: Read open/closed pull requests using `python .agents/skills/read_github_prs/scripts/read_prs.py --limit <N>`.
- **create_github_issue**: Create a new issue using `python .agents/skills/create_github_issue/scripts/create_issue.py`
- **create_github_pr**: Open a new Pull Request using `python .agents/skills/create_github_pr/scripts/create_pr.py`
- **update_github_issue**: Modify an existing issue using `python .agents/skills/update_github_issue/scripts/update_issue.py`
- **update_github_pr**: Modify an existing Pull Request using `python .agents/skills/update_github_pr/scripts/update_pr.py`

Agents are encouraged to run these scripts to verify current bugs and project status before directly modifying files, or to check out PRs before pushing similar code. If more details are needed, use `gh issue view` or `gh pr view`.
