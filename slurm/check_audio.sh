#!/bin/bash

check_audio_stream() {
    local file="$1"
    ffmpeg_output=$(ffmpeg -i "$file" 2>&1)
    if echo "$ffmpeg_output" | grep -q "Stream.*Audio"; then
        :
    else
        echo "$file"
    fi
}

while IFS= read -r file; do
    check_audio_stream "$file"
done
