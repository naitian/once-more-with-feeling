#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file=$1

if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found."
    exit 1
fi

while IFS= read -r filepath; do
    if [ -e "$filepath" ]; then
        :
    else
        echo $filepath
    fi
done < "$input_file"
