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

# Include any dependencies generated for this target.
include examples/CMakeFiles/benchmarks_all_unity.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/benchmarks_all_unity.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/benchmarks_all_unity.dir/flags.make

examples/cotire/benchmarks_all_CXX_unity.cxx: examples/benchmarks_all_CXX_Release_cotire.cmake
	$(CMAKE_COMMAND) -E cmake_progress_report /tiny_cnn/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Generating CXX unity source examples/cotire/benchmarks_all_CXX_unity.cxx"
	cd /tiny_cnn/examples && /usr/bin/cmake -DCOTIRE_BUILD_TYPE:STRING=Release -DCOTIRE_VERBOSE:BOOL=$(VERBOSE) -P /tiny_cnn/cmake/Modules/cotire.cmake unity /tiny_cnn/examples/benchmarks_all_CXX_Release_cotire.cmake /tiny_cnn/examples/cotire/benchmarks_all_CXX_unity.cxx

examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o: examples/CMakeFiles/benchmarks_all_unity.dir/flags.make
examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o: examples/cotire/benchmarks_all_CXX_unity.cxx
	$(CMAKE_COMMAND) -E cmake_progress_report /tiny_cnn/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o"
	cd /tiny_cnn/examples && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o -c /tiny_cnn/examples/cotire/benchmarks_all_CXX_unity.cxx

examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.i"
	cd /tiny_cnn/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /tiny_cnn/examples/cotire/benchmarks_all_CXX_unity.cxx > CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.i

examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.s"
	cd /tiny_cnn/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /tiny_cnn/examples/cotire/benchmarks_all_CXX_unity.cxx -o CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.s

examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o.requires:
.PHONY : examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o.requires

examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o.provides: examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o.requires
	$(MAKE) -f examples/CMakeFiles/benchmarks_all_unity.dir/build.make examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o.provides.build
.PHONY : examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o.provides

examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o.provides.build: examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o

# Object files for target benchmarks_all_unity
benchmarks_all_unity_OBJECTS = \
"CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o"

# External object files for target benchmarks_all_unity
benchmarks_all_unity_EXTERNAL_OBJECTS =

examples/benchmarks_all: examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o
examples/benchmarks_all: examples/CMakeFiles/benchmarks_all_unity.dir/build.make
examples/benchmarks_all: examples/CMakeFiles/benchmarks_all_unity.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable benchmarks_all"
	cd /tiny_cnn/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/benchmarks_all_unity.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/benchmarks_all_unity.dir/build: examples/benchmarks_all
.PHONY : examples/CMakeFiles/benchmarks_all_unity.dir/build

examples/CMakeFiles/benchmarks_all_unity.dir/requires: examples/CMakeFiles/benchmarks_all_unity.dir/cotire/benchmarks_all_CXX_unity.cxx.o.requires
.PHONY : examples/CMakeFiles/benchmarks_all_unity.dir/requires

examples/CMakeFiles/benchmarks_all_unity.dir/clean:
	cd /tiny_cnn/examples && $(CMAKE_COMMAND) -P CMakeFiles/benchmarks_all_unity.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/benchmarks_all_unity.dir/clean

examples/CMakeFiles/benchmarks_all_unity.dir/depend: examples/cotire/benchmarks_all_CXX_unity.cxx
	cd /tiny_cnn && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tiny_cnn /tiny_cnn/examples /tiny_cnn /tiny_cnn/examples /tiny_cnn/examples/CMakeFiles/benchmarks_all_unity.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/benchmarks_all_unity.dir/depend

