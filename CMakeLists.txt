cmake_minimum_required(VERSION 3.10)

# Set the project name
project(Micrograd)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define an option to switch between main and test
option(USE_TEST "Use the test source file" OFF)

# Conditionally add the source file based on the option
if(USE_TEST)
    message(STATUS "Building with test files")
    add_executable(Micrograd test_Value.cpp)
else()
    add_executable(Micrograd main.cpp)
endif()

# Include directories
target_include_directories(Micrograd PRIVATE .)
