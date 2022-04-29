#--------------------------------------------------------
# CMake for generating RefinerLib Testing Suite
#
# @author: Wild Boar
#
# Date Created: 2022-04-07
#--------------------------------------------------------

# Add Google Test to the test group
find_package(GTest REQUIRED)

# Setup the includes
include_directories("../")

# Create the executable
add_executable (RefinerTests
    Helpers/PolyProblem.cpp
	Tests/REngine_Tests.cpp
)

# Add link libraries
target_link_libraries(RefinerTests RefinerLib ${OpenCV_LIBS} GTest::Main)

# Find the associated unit tests
gtest_discover_tests(RefinerTests)