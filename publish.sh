#!/bin/bash
# Quick publish script for FrameWise

set -e  # Exit on error

echo "🎬 FrameWise Publishing Script"
echo "=============================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this script from the project root."
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo "📋 Checking prerequisites..."
if ! command_exists poetry; then
    echo "❌ Poetry not found. Install it with: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

if ! command_exists twine; then
    echo "⚠️  Twine not found. Installing..."
    pip install twine
fi

echo "✅ All prerequisites met"
echo ""

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info
echo "✅ Cleaned"
echo ""

# Build the package
echo "🔨 Building package..."
poetry build
echo "✅ Package built"
echo ""

# Check the distribution
echo "🔍 Checking distribution..."
twine check dist/*
echo "✅ Distribution check passed"
echo ""

# Ask user what to do
echo "📦 Package is ready!"
echo ""
echo "What would you like to do?"
echo "1) Upload to TestPyPI (recommended first)"
echo "2) Upload to PyPI (production)"
echo "3) Exit"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "📤 Uploading to TestPyPI..."
        twine upload --repository testpypi dist/*
        echo ""
        echo "✅ Uploaded to TestPyPI!"
        echo ""
        echo "Test installation with:"
        echo "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ framewise"
        ;;
    2)
        echo ""
        read -p "⚠️  Are you sure you want to upload to PyPI? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            echo "📤 Uploading to PyPI..."
            twine upload dist/*
            echo ""
            echo "✅ Uploaded to PyPI!"
            echo ""
            echo "🎉 Package is now live at: https://pypi.org/project/framewise/"
            echo ""
            echo "Test installation with:"
            echo "pip install framewise"
        else
            echo "❌ Upload cancelled"
        fi
        ;;
    3)
        echo "👋 Exiting without upload"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "📚 For more details, see PUBLISHING.md"
