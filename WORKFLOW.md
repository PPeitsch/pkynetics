# Pkynetics Development Workflow

This document describes the standard development workflow for contributing to Pkynetics.

---

## Quick Reference

| Task | Command |
|------|---------|
| Format code | `black src/pkynetics tests && isort src/pkynetics tests` |
| Check formatting | `black --check src/pkynetics tests && isort --check-only src/pkynetics tests` |
| Type check | `mypy` |
| Run tests | `pytest tests/ -v` |
| Run tests with coverage | `pytest tests/ -v --cov=src/pkynetics --cov-report=term-missing` |
| Install dev | `pip install -e .[dev]` |

---

## 1. Environment Setup

### First Time Setup

```bash
# Clone the repository
git clone https://github.com/PPeitsch/pkynetics.git
cd pkynetics

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install in development mode
pip install -e .[dev]
```

### Daily Workflow

```bash
# Activate environment
source .venv/bin/activate  # or Windows equivalent

# Pull latest changes
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

---

## 2. Development Cycle

### Writing Code

1. **Make your changes** in `src/pkynetics/`
2. **Add type hints** to all new functions
3. **Write tests** in `tests/` for new functionality
4. **Update docstrings** using NumPy format

### Code Quality Checks

Run these before every commit:

```bash
# Format code
black src/pkynetics tests
isort src/pkynetics tests

# Type checking
mypy

# Run tests
pytest tests/ -v
```

### Pre-Commit Checklist

- [ ] Code formatted with black
- [ ] Imports sorted with isort
- [ ] mypy passes without errors
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Docstrings added/updated
- [ ] CHANGELOG.md updated (if user-facing change)

---

## 3. Commit Guidelines

### Commit Message Format

```
type: brief description

- Detailed change 1
- Detailed change 2
```

### Commit Types

| Type | Use For |
|------|---------|
| `feat:` | New features, files, or functionality |
| `fix:` | Bug fixes |
| `refactor:` | Modifications to existing functionality |
| `chore:` | Removing code, maintenance |
| `docs:` | Documentation only changes |
| `test:` | Test additions or modifications |
| `release:` | Version releases |

### Examples

```bash
git commit -m "feat: implement Kissinger method for activation energy calculation"
git commit -m "fix: correct temperature unit conversion in DSC importer"
git commit -m "docs: update README with installation instructions"
```

---

## 4. Pull Request Process

### Before Creating PR

1. Ensure your branch is up to date:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. Run full quality check:
   ```bash
   black --check src/pkynetics tests
   isort --check-only src/pkynetics tests
   mypy
   pytest tests/ -v --cov=src/pkynetics
   ```

3. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

### PR Checklist

Use the PR template provided. Ensure:
- [ ] Description explains the changes
- [ ] Type of change is indicated
- [ ] All CI checks pass
- [ ] Tests added for new features
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated

---

## 5. Release Workflow

### Version Bump

1. Update version in `src/pkynetics/__about__.py`:
   ```python
   __version__ = "X.Y.Z"
   ```

2. Update version in `docs/conf.py`:
   ```python
   release = "X.Y.Z"
   ```

3. Update `CHANGELOG.md`:
   ```markdown
   ## [vX.Y.Z] - YYYY-MM-DD

   ### Added
   - New feature description

   ### Fixed
   - Bug fix description
   ```

### Create Release

```bash
# Commit version changes
git add -A
git commit -m "release: version X.Y.Z"

# Push to main
git push origin main

# Create and push tag
git tag vX.Y.Z
git push origin vX.Y.Z
```

The CI/CD pipeline will automatically:
1. Run all quality checks
2. Run tests on Python 3.9, 3.10, 3.11
3. Build and publish to PyPI

---

## 6. Troubleshooting

### Common Issues

#### Black/isort conflicts
```bash
# Run both in correct order
isort src/pkynetics tests
black src/pkynetics tests
```

#### mypy errors with scientific libraries
These are ignored in `pyproject.toml` via `[[tool.mypy.overrides]]`

#### Test failures due to numerical precision
Use `np.allclose()` or `pytest.approx()` for floating-point comparisons

### Getting Help

- Check existing issues on GitHub
- Review the documentation at ReadTheDocs
- Open a new issue with detailed description

---

## 7. Directory Reference

```
src/pkynetics/
├── __about__.py              # Version info
├── __init__.py               # Package initialization
├── data_import/              # TGA, DSC, Dilatometry importers
├── data_preprocessing/       # Data cleaning utilities
├── model_fitting_methods/    # Kissinger, JMAK, etc.
├── model_free_methods/       # Friedman, KAS, OFW
├── result_visualization/     # Plotting utilities
├── synthetic_data/           # Test data generators
└── technique_analysis/       # DSC, Dilatometry analysis
    └── dsc/                  # DSC-specific modules
```
