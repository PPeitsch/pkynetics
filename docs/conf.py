# Configuration file for the Sphinx documentation builder.

project = "Pkynetics"
copyright = "2024, Pablo Peitsch"
author = "Pablo Peitsch"
release = "0.6.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    # Lets docs/changelog.md include CHANGELOG.md directly, so the published
    # changelog cannot drift from the repository's (it had stopped at v0.3.6)
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
