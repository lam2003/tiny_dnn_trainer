# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /tiny_cnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /tiny_cnn

# Utility rule file for all_unity.

# Include the progress variables for this target.
include examples/CMakeFiles/all_unity.dir/progress.make

all_unity: examples/CMakeFiles/all_unity.dir/build.make
.PHONY : all_unity

# Rule to build all files generated by this target.
examples/CMakeFiles/all_unity.dir/build: all_unity
.PHONY : examples/CMakeFiles/all_unity.dir/build

examples/CMakeFiles/all_unity.dir/clean:
	cd /tiny_cnn/examples && $(CMAKE_COMMAND) -P CMakeFiles/all_unity.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/all_unity.dir/clean

examples/CMakeFiles/all_unity.dir/depend:
	cd /tiny_cnn && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tiny_cnn /tiny_cnn/examples /tiny_cnn /tiny_cnn/examples /tiny_cnn/examples/CMakeFiles/all_unity.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/all_unity.dir/depend

