#!/bin/bash

# Create build directory if it doesn't exist
[ ! -d "build" ] && mkdir build

# Navigate to the build directory
cd build

# Configure CMake based on the argument
if [ "$1" == "test" ]; then
    cmake -DUSE_TEST=ON ..
else
    cmake -DUSE_TEST=OFF ..
fi

# Build the project
cmake --build .

# Check if the build was successful
if [ $? -eq 0 ]; then
    # Run the executable
    ./Micrograd
else
    echo "Build failed, not running the executable."
fi
