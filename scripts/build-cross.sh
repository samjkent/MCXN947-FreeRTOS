#!/bin/bash

# Check if the 'build' directory exists and is empty
if [ -d "build" ] && [ -z "$(ls -A build)" ]; then
    echo "Build folder is empty. Running meson setup..."
    meson setup build --cross-file=arm-gcc.txt
elif [ ! -d "build" ]; then
    echo "Build folder does not exist. Creating and setting up meson..."
    meson setup build --cross-file=arm-gcc.txt
else
    echo "Build folder is not empty. Skipping meson setup."
fi

# Always run ninja
cd build && ninja

