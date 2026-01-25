# How to Publish to PyPI (Python Package Index)

**PyPI** is the official repository for Python software. Publishing here allows anyone to install Farnsworth using:

```bash
pip install farnsworth-ai
```

## Prerequisites

1.  **Create an account** on [TestPyPI](https://test.pypi.org/account/register/) (for testing) and [PyPI](https://pypi.org/account/register/) (for production).
2.  **Enable 2FA** on your PyPI account (required).
3.  **Create an API Token** in your account settings.

## Steps to Publish

### 1. Install Build Tools

**Windows:**
```bash
py -m pip install build twine
```

**Mac/Linux:**
```bash
python3 -m pip install build twine
```

### 2. Build the Package

This creates the distribution files in `dist/`.

```bash
python -m build
```

### 3. Check the Build

Ensure the descriptions look right.

```bash
twine check dist/*
```

### 4. Upload to PyPI

```bash
twine upload dist/*
```

You will be prompted for your username (use `__token__`) and your API token as the password.

## After Publishing

Users can now install your tool universally:

```bash
pip install farnsworth-ai
farnsworth-server  # Runs the server
```
