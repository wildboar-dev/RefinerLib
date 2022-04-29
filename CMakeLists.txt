#--------------------------------------------------------
# Top-Level: Refiner
#
# @author: Wild Boar
#
# @Date Created: 2022-04-07
#--------------------------------------------------------

# Create the main application
cmake_minimum_required(VERSION 3.0.0)
project(Refiner VERSION 0.1.0)

# Enable Testing
enable_testing()

# Set the correct version of C++
set(CMAKE_CXX_STANDARD 17)

# Add opencv to the solution
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Add the available subfolders
add_subdirectory(RefinerLib)
add_subdirectory(RefinerTests)

