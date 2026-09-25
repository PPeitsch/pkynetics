# AGENTS.md — Pkynetics

> **CRITICAL RULE**: never run `git commit` or `git push` without first running the full
> quality pipeline — `black`, `isort`, `mypy`, `pytest`, `sphinx-build` — successfully.

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

Run all five before every commit. This is what CI runs.

```bash
black --check . && isort --check-only . && mypy src && pytest --cov \
  && sphinx-build -b html -W docs docs/_build/html
```

`black` and `isort` apply with `black .` and `isort .` (line length 88, isort profile
`black`). `mypy` is in **strict** mode.

**The docs build is part of the pipeline, not an afterthought.** `-W` turns Sphinx
warnings into errors, so a malformed docstring, a duplicate object description or a
broken reference fails the Documentation job — and those come from ordinary source
changes, not from touching `docs/`. Leaving it out means finding out from CI: it is the
one step here that has actually caught a push. The toolchain comes with `pip install
-e .[dev]`, the output goes to `docs/_build/`, which is gitignored, and it takes a few
seconds.

CI additionally builds the sdist and runs the tests against the installed wheel, on
Python 3.10–3.13.

## Conventions

- **Commits** — Conventional Commits: `type: description`, with
  `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `release`.
- **Type hints** on every new function and method; **NumPy-style docstrings**.
- **Tests** for new functionality.
- **`CHANGELOG.md`** updated for any user-facing change
  ([Keep a Changelog](https://keepachangelog.com/en/1.0.0/)).
- **88-character lines** (black default).

## Validating a numerical method

Any change that alters the numbers the package produces goes through this loop before it
is merged. That covers a new or changed smoother, derivative, detector, fit or threshold.
Tests prove the code does what it was written to do; they do not prove it gives the right
answer on real data, and several past fixes (issue #94, PR #101, issue #26) only surfaced
when someone looked at a plot.

1. **Script.** Run the method on the shipped example data, all relevant runs (e.g. both
   Zry-4 dilatometry runs, heating and cooling), and on synthetic data with a known
   answer. Compare against the current behaviour, not just in isolation. Sweep every
   parameter the result depends on over a **fine grid**: a coarse one (issue #112 sampled
   the window at 30 and 40 K) can miss a collapse sitting in between.
2. **Plots.** Draw the result so it can be judged by eye: the curve with the limits or
   fit overlaid, and the result against each swept parameter. Decisions are made on the
   plots, not on the tables alone.
3. **Review.** The maintainer reads the plots and says what looks wrong from domain
   knowledge ("the start should be ~830", "why does it fail at 7-9"). Wait for that
   review; do not merge a method change on your own reading of the plots.
4. **Answer the review with evidence.** Check each point against the numbers, find the
   cause, and say whether it is the method, a parameter, or something else. In issue #112
   the collapses blamed on the derivative estimator turned out to be the detector (issue
   #115). Then propose a correction or an improvement, or explain why it cannot be
   improved. Measure the proposal the same way (back to 1) until the plots are agreed.
5. **Code.** Only then write the change. Turn the cases the loop found into regression
   tests, and record the numbers in the issue or PR.
6. **Record it in `validation.yaml`.** Every time this loop runs, whatever its outcome,
   update the method's entry in the same PR: `status`, `date`, `commit`, `data`, `how`,
   `evidence` and `notes`. A validation that is not recorded will be done again.

The judging criteria are error against a known answer on synthetic data, and **stability
against the parameters** on real data: a result that holds across a range of settings,
not one that is right at the default. Linearity or goodness of fit alone has been
misleading (issue #26).

Keep the scripts and plots out of the repository; they are working material. Promote
one to `examples/` only if it teaches something a user needs.

### The validation registry

`validation.yaml`, at the repository root, lists every method that produces numbers and
where it stands:

| `status` | meaning |
|---|---|
| `validated` | went through the loop above and the maintainer agreed it is correct |
| `issues` | works, with known problems linked in `evidence` |
| `failed` | gives wrong results; do not rely on it |
| `pending` | not validated yet |

The registry is the single source of truth. It is not repeated in docstrings, where it
would drift. `tests/test_validation_registry.py` keeps it honest: every `target` must
import, every non-pending entry must say when, against which commit, on what data, how,
and where the evidence is.

**Do not validate again what is validated and unchanged.** Before starting the loop on
a method, run:

```bash
python tests/test_validation_registry.py
```

It prints each entry's status and flags the validated ones whose source changed since
their `commit`: the target's own file plus any listed in `files`. List there the helpers
the result depends on. A flagged entry needs the loop again; an unflagged `validated`
one does not. New methods get an entry, `pending` until validated.

Once the features in flight are finished and stable, every `pending` entry goes through
the loop.


## Release

1. Tests green locally.
2. Bump `src/pkynetics/__about__.py` **and** `docs/conf.py` — the version is not yet
   centralised, so they drift unless updated together.
3. `CHANGELOG.md` with the release date.
4. `release: version X.Y.Z`, then `git tag vX.Y.Z && git push origin vX.Y.Z`.
5. CI publishes to PyPI from the distributions the Package job tested.

## Docs

Docs are Sphinx (`docs/`, Napoleon, NumPy docstrings), published to ReadTheDocs.
