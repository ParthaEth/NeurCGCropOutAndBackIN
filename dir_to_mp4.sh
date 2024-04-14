#!/bin/bash

# Set the directory containing the frames
frame_dir="./NeurCG_fullbody_kf"

# Initialize a counter
counter=0

# Loop through each file in numerical order, sorted by their existing numbers
cd $frame_dir
for frame in $(ls frame_*.png | sort -V); do
    # Format new filename with counter, ensuring four-digit numbering with leading zeros
    new_name=$(printf "frame_%04d.png" $counter)

    # Move (rename) the file to the new name
    mv "$frame" "$new_name"

    # Increment the counter
    ((counter++))
    echo $frame
done

echo "Renaming complete. Files are renamed sequentially from frame_0000.png onwards."

ffmpeg -framerate 25 -i "$frame_dir/frame_%04d.png" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "$frame_dir/../NeurCG_full_body_kf.mp4"
