# Publishing FrameWise to PyPI

This guide walks you through publishing FrameWise to PyPI (Python Package Index).

## Prerequisites

1. **PyPI Account**: Create accounts on both:
   - [PyPI](https://pypi.org/account/register/) (production)
   - [TestPyPI](https://test.pypi.org/account/register/) (testing)

2. **Install Required Tools**:
```bash
pip install --upgrade pip
pip install --upgrade build twine
```

3. **Configure PyPI Credentials**:
Create `~/.pypirc` file:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TEST_API_TOKEN_HERE
```

To get API tokens:
- PyPI: https://pypi.org/manage/account/token/
- TestPyPI: https://test.pypi.org/manage/account/token/

## Step-by-Step Publishing Process

### 1. Clean Previous Builds
```bash
rm -rf dist/ build/ *.egg-info
```

### 2. Update Version Number
Edit `pyproject.toml` and `framewise/__init__.py`:
```python
__version__ = "0.1.0"  # Update this
```

### 3. Build the Package
```bash
# Using Poetry (recommended)
poetry build

# Or using build tool
python -m build
```

This creates:
- `dist/framewise-0.1.0.tar.gz` (source distribution)
- `dist/framewise-0.1.0-py3-none-any.whl` (wheel distribution)

### 4. Test the Package Locally
```bash
# Install in a virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from local build
pip install dist/framewise-0.1.0-py3-none-any.whl

# Test it works
python -c "from framewise import TranscriptExtractor; print('Success!')"
framewise version

# Clean up
deactivate
rm -rf test_env
```

### 5. Upload to TestPyPI (Optional but Recommended)
```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ framewise
```

Note: `--extra-index-url` is needed because TestPyPI doesn't have all dependencies.

### 6. Upload to PyPI (Production)
```bash
# Upload to PyPI
twine upload dist/*

# Or with Poetry
poetry publish
```

### 7. Verify Installation
```bash
# Install from PyPI
pip install framewise

# Test it
python -c "from framewise import TranscriptExtractor; print('Success!')"
```

## Version Management

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Example version progression:
- `0.1.0` - Initial release
- `0.1.1` - Bug fix
- `0.2.0` - New feature
- `1.0.0` - Stable release

## Git Tagging

Tag releases in Git:
```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

## Automated Publishing with GitHub Actions

Create `.github/workflows/publish.yml`:
```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      
      - name: Build package
        run: poetry build
      
      - name: Publish to PyPI
        env:
          POETRY_PYPI_TOKEN_PYPI: ${{ secrets.PYPI_API_TOKEN }}
        run: poetry publish
```

Add your PyPI API token to GitHub Secrets:
1. Go to repository Settings → Secrets → Actions
2. Add `PYPI_API_TOKEN` with your PyPI token

## Checklist Before Publishing

- [ ] All tests pass: `pytest tests/`
- [ ] Version number updated in `pyproject.toml` and `framewise/__init__.py`
- [ ] CHANGELOG.md updated (if you have one)
- [ ] README.md is up to date
- [ ] All dependencies are correctly listed in `pyproject.toml`
- [ ] Package builds successfully: `poetry build`
- [ ] Package installs locally: `pip install dist/*.whl`
- [ ] Git changes committed and pushed
- [ ] Git tag created for the version

## Common Issues

### Issue: "File already exists"
**Solution**: You can't re-upload the same version. Increment the version number.

### Issue: Missing dependencies
**Solution**: Ensure all dependencies are in `pyproject.toml` under `[tool.poetry.dependencies]`

### Issue: Import errors after installation
**Solution**: Check that `__init__.py` files exist in all package directories

### Issue: README not showing on PyPI
**Solution**: Ensure `readme = "README.md"` is in `pyproject.toml`

## Post-Publication

1. **Verify on PyPI**: Check https://pypi.org/project/framewise/
2. **Test Installation**: `pip install framewise` in a fresh environment
3. **Update Documentation**: Add installation instructions to README
4. **Announce**: Share on social media, relevant forums, etc.

## Useful Commands

```bash
# Check package metadata
poetry check

# Show package info
poetry show framewise

# List all files that will be included
poetry build -vvv

# Check distribution
twine check dist/*

# View package on PyPI
open https://pypi.org/project/framewise/
```

## Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [PyPI Help](https://pypi.org/help/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
