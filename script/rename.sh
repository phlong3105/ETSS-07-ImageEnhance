#!/bin/bash

# Main
clear
echo "$HOSTNAME"

# Define the directory where you want to perform the recursive renaming
directory="/media/longpham/hdd_01/30_areas/family/photos"


# Validate the directory
if [ ! -d "$directory" ]; then
    echo "Error: '$directory' is not a valid directory."
    exit 1
fi

# Function to normalize names (lowercase, replace ' ' and '-' with '_', and reduce '__' to '_')
normalize_name() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr '-' '_' | sed 's/__\+/_/g'
}

# Process both files and directories in one pass
find "$directory" -depth | while IFS= read -r path; do
    # Skip if it's the target directory itself
    if [ "$path" = "$directory" ]; then
        continue
    fi

    # Extract directory and base name
    dir=$(dirname "$path")
    old_name=$(basename "$path")

    # Generate new normalized name
    new_name=$(normalize_name "$old_name")

    # Rename if the name has changed
    if [ "$old_name" != "$new_name" ]; then
        mv -v "$dir/$old_name" "$dir/$new_name"
    fi
done

echo "Renaming process completed."
exit 0
