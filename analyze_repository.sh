#!/bin/bash

# Safe Repository Analysis and Cleanup Script
# This script analyzes files before any deletion and backs up important logic

set -e

echo "🔍 Starting safe repository analysis..."

# Create backup directory
BACKUP_DIR="./backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📋 Analyzing suspicious files for important logic..."

# Function to analyze a file and determine its type and importance
analyze_file() {
    local file="$1"
    local backup_needed=false
    local file_type=""
    local importance=""
    
    if [ ! -f "$file" ]; then
        return
    fi
    
    # Check file type
    if head -n 1 "$file" | grep -q "%!PS-Adobe"; then
        file_type="PostScript/Image"
        importance="low"
    elif head -n 1 "$file" | grep -q "#!/usr/bin/env node"; then
        file_type="Node.js CLI script"
        importance="high"
        backup_needed=true
    elif head -n 1 "$file" | grep -q "#!/"; then
        file_type="Shell/Script"
        importance="medium"
        backup_needed=true
    elif grep -q "import\|require\|function\|class\|def " "$file" 2>/dev/null; then
        file_type="Code file"
        importance="high"
        backup_needed=true
    else
        file_type="Unknown/Binary"
        importance="medium"
        backup_needed=true
    fi
    
    echo "  📄 $file: $file_type ($importance importance)"
    
    if [ "$backup_needed" = true ]; then
        echo "    💾 Backing up to $BACKUP_DIR/"
        cp "$file" "$BACKUP_DIR/"
    fi
    
    # Return the importance level
    echo "$importance"
}

# List of suspicious files to analyze
SUSPICIOUS_FILES=(
    "asyncio"
    "claude-flow"
    "claude-flow.bat"
    "claude-flow.ps1"
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

echo ""
echo "🔍 Analyzing each suspicious file..."

# Analyze each file
declare -A file_analysis
for file in "${SUSPICIOUS_FILES[@]}"; do
    if [ -f "$file" ]; then
        importance=$(analyze_file "$file")
        file_analysis["$file"]="$importance"
    fi
done

echo ""
echo "📊 Analysis Summary:"
echo "==================="

# Categorize files by importance
high_importance=()
medium_importance=()
low_importance=()

for file in "${!file_analysis[@]}"; do
    case "${file_analysis[$file]}" in
        "high")
            high_importance+=("$file")
            ;;
        "medium")
            medium_importance+=("$file")
            ;;
        "low")
            low_importance+=("$file")
            ;;
    esac
done

echo ""
echo "🔴 HIGH IMPORTANCE (DO NOT DELETE - Contains important logic):"
for file in "${high_importance[@]}"; do
    echo "  - $file"
done

echo ""
echo "🟡 MEDIUM IMPORTANCE (Review before deletion):"
for file in "${medium_importance[@]}"; do
    echo "  - $file"
done

echo ""
echo "🟢 LOW IMPORTANCE (Safe to delete - PostScript/Image files):"
for file in "${low_importance[@]}"; do
    echo "  - $file"
done

echo ""
echo "📁 Backup created at: $BACKUP_DIR"
echo ""

# Check if any high importance files exist
if [ ${#high_importance[@]} -gt 0 ]; then
    echo "⚠️  WARNING: High importance files detected!"
    echo "   These files contain important logic and should NOT be deleted."
    echo "   Please review them manually and decide how to handle them."
    echo ""
    
    for file in "${high_importance[@]}"; do
        echo "🔍 Detailed analysis of $file:"
        echo "   File size: $(wc -c < "$file") bytes"
        echo "   Line count: $(wc -l < "$file") lines"
        echo "   First few lines:"
        head -n 5 "$file" | sed 's/^/     /'
        echo ""
    done
fi

echo "✅ Analysis complete. Review the above output before proceeding with any deletions."
echo ""
echo "Recommended next steps:"
echo "1. Review backed up files in $BACKUP_DIR"
echo "2. For HIGH importance files, determine if they belong in the project"
echo "3. For MEDIUM importance files, review their content"
echo "4. Only delete LOW importance files (PostScript/Image files)"
echo "5. Update .gitignore to prevent tracking unwanted files in the future"
