#!/bin/bash

# Create build directory if it doesn't exist
[ ! -d "build" ] && mkdir build

# Navigate to the build directory
cd build

# Run CMake and build
cmake .. && cmake --build .
