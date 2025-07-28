#!/bin/bash

# Repository Cleanup Script
# This script removes unwanted files and fixes the repository structure

set -e

echo "🧹 Starting repository cleanup..."

# Remove PostScript files that are incorrectly named as Python modules
echo "Removing PostScript files masquerading as Python modules..."
FILES_TO_REMOVE=(
    "asyncio"
    "claude-flow"
    "json"
    "os"
    "redis"
    "requests"
    "st"
    "sys"
    "time"
    "uvicorn"
    "websockets"
)

for file in "${FILES_TO_REMOVE[@]}"; do
    if [ -f "$file" ]; then
        echo "  Removing: $file"
        rm -f "$file"
    fi
done

# Remove batch files that shouldn't be tracked
echo "Removing unnecessary batch files..."
rm -f claude-flow.bat
rm -f webui.bat

# Remove generated files and temporary data
echo "Removing generated and temporary files..."
rm -f final-*.mp4
rm -f test_*.mp4
rm -f deployment.log
rm -f progress.db
rm -f PROGRESS.db
rm -f PROGRESS.sql

# Remove development tool directories if they exist
echo "Removing development tool directories..."
rm -rf .roo
rm -rf .aider.*
rm -rf .memory
rm -rf .claude
rm -rf .swarm
rm -rf .hive-mind

# Remove cache directories
echo "Removing cache directories..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
find . -name "*.pyd" -delete 2>/dev/null || true

# Remove pytest cache
rm -rf .pytest_cache

# Clean up storage directory but keep structure
echo "Cleaning storage directory..."
if [ -d "storage" ]; then
    find storage -type f -name "*.mp4" -delete 2>/dev/null || true
    find storage -type f -name "*.wav" -delete 2>/dev/null || true
    find storage -type f -name "*.mp3" -delete 2>/dev/null || true
fi

# Remove test files in root
echo "Removing root-level test files..."
rm -f test_*.py

# If git is available, remove from git tracking
if command -v git &> /dev/null && [ -d ".git" ]; then
    echo "Removing files from Git tracking..."
    
    # Remove files from git cache
    for file in "${FILES_TO_REMOVE[@]}"; do
        git rm --cached "$file" 2>/dev/null || true
    done
    
    git rm --cached claude-flow.bat 2>/dev/null || true
    git rm --cached webui.bat 2>/dev/null || true
    git rm --cached final-*.mp4 2>/dev/null || true
    git rm --cached test_*.py 2>/dev/null || true
    git rm --cached deployment.log 2>/dev/null || true
    git rm --cached progress.db 2>/dev/null || true
    git rm --cached PROGRESS.db 2>/dev/null || true
    git rm --cached PROGRESS.sql 2>/dev/null || true
    
    # Remove directories from git cache
    git rm -r --cached .roo 2>/dev/null || true
    git rm -r --cached .memory 2>/dev/null || true
    git rm -r --cached .claude 2>/dev/null || true
    git rm -r --cached .swarm 2>/dev/null || true
    git rm -r --cached .hive-mind 2>/dev/null || true
    git rm -r --cached __pycache__ 2>/dev/null || true
    
    echo "✅ Git cleanup completed"
else
    echo "⚠️  Git not available or not a git repository"
fi

echo "✨ Repository cleanup completed!"
echo ""
echo "Summary of actions taken:"
echo "- Removed PostScript files with module names"
echo "- Removed unnecessary batch files"
echo "- Removed generated video files"
echo "- Removed cache and temporary directories"
echo "- Removed development tool directories"
echo "- Updated git tracking (if available)"
echo ""
echo "Next steps:"
echo "1. Review the changes: git status"
echo "2. Commit the cleanup: git add -A && git commit -m 'chore: clean up repository and remove unnecessary files'"
echo "3. Push to remote: git push"
