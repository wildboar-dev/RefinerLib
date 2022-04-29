#--------------------------------------------------------
# CMake for generating the RefinerLib
#
# @author: Wild Boar
#
# Date Created: 2022-04-07
#--------------------------------------------------------

# Setup include directories
include_directories("../")

# Create Library
add_library (RefinerLib STATIC
	REngine.cpp
)