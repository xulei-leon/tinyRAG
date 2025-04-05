#!/bin/bash

# Check if exactly two arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_directory> <output_directory>"
  exit 1
fi

# Get the input and output directories
input_dir="$1"
output_dir="$2"

# Ensure the input directory exists and is a directory
if [ ! -d "$input_dir" ]; then
  echo "Error: Input directory '$input_dir' does not exist or is not a directory."
  exit 1
fi

# Ensure the output directory exists, create it if it doesn't
if [ ! -d "$output_dir" ]; then
  echo "Hint: Output directory '$output_dir' does not exist, creating it..."
  mkdir -p "$output_dir"
fi

# Recursively find all files in the input directory using the find command
find "$input_dir" -type f -print0 | while IFS= read -r -d $'\0' input_file; do
  # Get the relative path to the input directory
  relative_path="${input_file#"$input_dir/"}"

  # Construct the output file path
  output_file="${output_dir}/${relative_path%.*}.txt"

  # Construct the output directory path
  output_dir_path=$(dirname "$output_file")

  # Create the output directory if it doesn't exist
  mkdir -p "$output_dir_path"

  # Convert the file using pandoc
  echo "Converting: '$input_file' -> '$output_file'"
  pandoc "$input_file" -o "$output_file"
  if [ "$?" -ne 0 ]; then
    echo "Warning: Conversion of file '$input_file' failed."
  fi
done

echo "Conversion complete."

exit 0