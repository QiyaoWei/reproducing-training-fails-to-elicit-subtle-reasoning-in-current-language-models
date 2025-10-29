#!/bin/bash
# Script to archive old documentation files
# All information is now consolidated in HPC_SETUP_README.md

# Create archive directory
ARCHIVE_DIR="./archived_docs"
mkdir -p "$ARCHIVE_DIR"

echo "=========================================="
echo "Archiving old documentation files"
echo "=========================================="
echo ""
echo "All information is now in: HPC_SETUP_README.md"
echo ""

# List of files to archive
FILES_TO_ARCHIVE=(
    "FINAL_FIX_SUMMARY.md"
    "FIX_NETWORK_ERROR.md"
    "MODEL_DOWNLOAD_README.md"
    "OFFLINE_MODE_FIX_SUMMARY.md"
    "QUICK_FIX_REFERENCE.md"
    "SETUP_SUMMARY.md"
    "TROUBLESHOOTING.md"
)

echo "The following files will be moved to $ARCHIVE_DIR:"
for file in "${FILES_TO_ARCHIVE[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done

echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for file in "${FILES_TO_ARCHIVE[@]}"; do
        if [ -f "$file" ]; then
            mv "$file" "$ARCHIVE_DIR/"
            echo "✓ Archived: $file"
        else
            echo "⊘ Not found: $file"
        fi
    done

    echo ""
    echo "=========================================="
    echo "Archive complete!"
    echo "=========================================="
    echo ""
    echo "Archived files are in: $ARCHIVE_DIR"
    echo "Active documentation: HPC_SETUP_README.md"
    echo ""
    echo "To restore archived files:"
    echo "  mv $ARCHIVE_DIR/*.md ."
else
    echo ""
    echo "Archive cancelled."
fi
