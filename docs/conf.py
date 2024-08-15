# Configuration file for the Sphinx documentation builder.

project = 'Pkynetics'
copyright = '2024, Pablo Peitsch'
author = 'Pablo Peitsch'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']
