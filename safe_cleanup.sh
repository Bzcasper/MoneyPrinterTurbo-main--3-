#!/bin/bash

# Safe Repository Cleanup Script
# Only removes verified PostScript files that are incorrectly named
# Preserves all important CLI tools and code files

set -e

echo "🧹 Starting SAFE repository cleanup..."
echo "📋 This script will ONLY remove PostScript files that are incorrectly named"
echo ""

# Create backup directory for safety
BACKUP_DIR="./postscript_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Function to verify a file is a PostScript file before deletion
verify_and_backup_postscript() {
    local file="$1"
    
    if [ ! -f "$file" ]; then
        echo "  ⚠️  File $file does not exist, skipping"
        return 1
    fi
    
    # Check if it's actually a PostScript file
    if head -n 1 "$file" | grep -q "%!PS-Adobe"; then
        echo "  ✅ Verified $file is a PostScript file"
        
        # Get file size for reporting
        size=$(wc -c < "$file")
        echo "     Size: $size bytes"
        
        # Backup before deletion (just in case)
        cp "$file" "$BACKUP_DIR/"
        echo "     Backed up to $BACKUP_DIR/"
        
        return 0
    else
        echo "  ❌ $file is NOT a PostScript file - keeping it safe!"
        return 1
    fi
}

# List of PostScript files to remove (verified to be PostScript)
POSTSCRIPT_FILES=(
    "asyncio"
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

echo "🔍 Verifying and removing PostScript files..."
echo ""

removed_count=0
total_size_saved=0

for file in "${POSTSCRIPT_FILES[@]}"; do
    echo "📄 Checking $file..."
    
    if verify_and_backup_postscript "$file"; then
        size=$(wc -c < "$file")
        rm "$file"
        echo "  🗑️  Removed $file"
        ((removed_count++))
        ((total_size_saved += size))
    fi
    echo ""
done

# Convert bytes to human readable format
if [ $total_size_saved -gt 1073741824 ]; then
    size_display="$(($total_size_saved / 1073741824))GB"
elif [ $total_size_saved -gt 1048576 ]; then
    size_display="$(($total_size_saved / 1048576))MB"
elif [ $total_size_saved -gt 1024 ]; then
    size_display="$(($total_size_saved / 1024))KB"
else
    size_display="${total_size_saved}B"
fi

echo "✨ Cleanup completed!"
echo "📊 Summary:"
echo "  - Files removed: $removed_count"
echo "  - Disk space saved: $size_display"
echo "  - Backup location: $BACKUP_DIR"
echo ""

# Verify important files are still present
echo "🔍 Verifying important files are still present..."
IMPORTANT_FILES=(
    "claude-flow"
    "claude-flow.bat"
    "claude-flow.ps1"
)

all_present=true
for file in "${IMPORTANT_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file - Present"
    else
        echo "  ❌ $file - MISSING!"
        all_present=false
    fi
done

if [ "$all_present" = true ]; then
    echo "  🎉 All important CLI files are preserved!"
else
    echo "  ⚠️  Some important files are missing - check backup directory"
fi

echo ""
echo "📝 Next steps:"
echo "1. Review the removed files backup in: $BACKUP_DIR"
echo "2. If everything looks good, you can remove the backup directory"
echo "3. Commit the changes: git add -A && git commit -m 'chore: remove PostScript files incorrectly named as Python modules'"
echo "4. The .gitignore has been updated to prevent tracking these file types in the future"
