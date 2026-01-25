# Contributing to Farnsworth

Thank you for your interest in contributing to Farnsworth! This document provides guidelines for contributing.

---

## Code of Conduct

- Be respectful and considerate
- Welcome newcomers and help them learn
- Focus on what is best for the community

---

## How Can I Contribute?

### Reporting Bugs

1. Check existing [GitHub Issues](https://github.com/timowhite88/Farnsworth/issues)
2. Include: OS, Python version, steps to reproduce, error messages

### Suggesting Features

1. Check the [ROADMAP.md](ROADMAP.md) first
2. Open an issue with the `feature-request` label

### Contributing Code

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/Farnsworth.git
cd Farnsworth

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a feature branch
git checkout -b feature/your-feature-name

# 5. Make changes and test
pytest tests/ -v

# 6. Commit with conventional commits
git commit -m "feat: add X feature"

# 7. Push and create PR
git push origin feature/your-feature-name
```

---

## Commit Message Format

```
<type>: <description>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- test: Adding tests
- refactor: Code restructuring
```

---

## Style Guidelines

- Use **Black** for formatting
- Use **isort** for imports
- Add type hints
- Write docstrings for public functions

---

## Testing

```bash
pytest tests/ -v                    # Run all tests
pytest tests/ --cov=farnsworth     # With coverage
```

---

## License

By contributing, you agree to the project's [LICENSE](LICENSE) terms.

---

*Thank you for helping make Farnsworth better!*
