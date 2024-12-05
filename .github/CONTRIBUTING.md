# Contributing Guidelines

*Pull requests, bug reports, and all other forms of contribution are welcomed and highly encouraged!* :octocat:

### Contents

- [Our Standards](#balance_scale-our-standards)
- [Getting Help](#thinking-getting-help)
- [Reporting Issues](#lady_beetle-reporting-issues)
- [Feature Proposals](#sparkles-feature-proposals)
- [Issue Triage](#microscope-issue-triage)
- [Pull Requests](#rocket-pull-requests)
- [Commit Guidelines](#pencil2-commit-guidelines)
- [Review Process](#eyes-review-process)
- [Code Style & Standards](#art-code-style--standards)
- [Developer's Certificate](#scroll-developers-certificate)

> **This guide sets clear expectations for everyone involved with the project so that we can improve it together while creating a welcoming space for everyone to participate.**

## :balance_scale: Our Standards

Please review our [Code of Conduct](CODE_OF_CONDUCT.md). It is in effect at all times. We expect it to be honored by everyone who contributes to this project. Disrespectful behavior will not be tolerated.

## :thinking: Getting Help

Before opening an issue, check the [documentation](https://pkynetics.readthedocs.io/). GitHub issues are for bugs and feature requests, not questions or debugging help.

## :lady_beetle: Reporting Issues

Before [creating an issue](https://help.github.com/en/github/managing-your-work-on-github/creating-an-issue), check if you are using the latest version of the project.

### :shield: Security Issues

Review our [Security Policy](SECURITY.md). **Do not** file public issues for security vulnerabilities.

### :bug: Bug Reports

Send detailed issue reports when you encounter bugs. A well-written bug report helps us immensely. :v:

Since you are most likely a developer, **provide a ticket that you would like to receive**.

- **Check the documentation** first
- **Avoid duplicates!** Search existing issues
- **Use reactions** instead of "+1" comments
- **Complete the template** fully
- **Use [GitHub-flavored Markdown](https://help.github.com/en/github/writing-on-github/basic-writing-and-formatting-syntax)**

## :sparkles: Feature Proposals

Feature proposals are welcome! We'll consider all requests but cannot guarantee acceptance. We aim to avoid [feature creep](https://en.wikipedia.org/wiki/Feature_creep).

- **Search first** for similar proposals
- **Complete the template** thoroughly
- **Be specific** about implementation details

## :microscope: Issue Triage

Help us by reproducing bug reports and providing additional information on issues. Any assistance in resolving issues is appreciated!

## :rocket: Pull Requests

We **love** pull requests! Before submitting significant changes, open an issue to discuss your proposal.

For accepted PRs, you'll be invited to become a project collaborator with **push access**. :tada:

*All contributions are licensed under the project's license.*

- **Keep it small.** One change per PR
- **Discuss big changes** first
- **Write clear code** over clever code
- **Follow project style**
- **Add tests**
- **Update documentation**
- **Update CHANGELOG**
- **Target default branch**
- **Resolve conflicts**
- **Fix CI failures**

## :pencil2: Commit Guidelines

Follow these [commit message guidelines](https://chris.beams.io/posts/git-commit/):

```
[TAG] Concise change description in 50 chars or less

More detailed explanation if needed. Wrap at 72 characters. Focus 
on WHY this change is being made rather than WHAT was changed.

The code itself shows what changed; use this space to explain 
motivation and context.

 - Bullet points welcome
 - Use hyphens or asterisks

Resolves: #123
See also: #456
```

## :eyes: Review Process

- **Focus on the code**, not the person
- **Don't take it personally**
- **Aim for quality**

## :art: Code Style & Standards

For Python code:
- Follow PEP 8
- Use type hints
- Write clear docstrings following NumPy style
- Use `black` formatter with line length 100
- Use `isort` for import sorting
- Pass `mypy` checks with strict settings
- Add appropriate tests with `pytest`

## :scroll: Developer's Certificate

*Developer's Certificate of Origin 1.1*

By contributing, you certify that:

> 1. Your contribution is your original work and you have the right to submit it under the project's license
> 2. Your contribution is based on previous work under an appropriate open source license
> 3. The contribution was provided to you by someone who certified 1 or 2
> 4. You understand this contribution is public and will be maintained indefinitely

## The Contributor's Token :key:

If you've read these guidelines (thank you!), include the ⚡ lightning emoji at the start of your pull requests and issues. You're awesome! :100:

---
*These contributing guidelines are based on the excellent work by Jesse Squires ([@jessesquires](https://github.com/jessesquires)).*
