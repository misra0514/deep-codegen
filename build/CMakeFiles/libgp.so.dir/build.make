# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /scratch/yguo25/files/nnsys/deep-codegen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /scratch/yguo25/files/nnsys/deep-codegen/build

# Utility rule file for libgp.so.

# Include any custom commands dependencies for this target.
include CMakeFiles/libgp.so.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/libgp.so.dir/progress.make

CMakeFiles/libgp.so: ../kernel.cu
CMakeFiles/libgp.so: ../kernel.h
CMakeFiles/libgp.so: ../op.h
CMakeFiles/libgp.so: ../Makefile
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/scratch/yguo25/files/nnsys/deep-codegen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Buidling libgp.so"
	cd /scratch/yguo25/files/nnsys/deep-codegen && /usr/bin/gmake

libgp.so: CMakeFiles/libgp.so
libgp.so: CMakeFiles/libgp.so.dir/build.make
.PHONY : libgp.so

# Rule to build all files generated by this target.
CMakeFiles/libgp.so.dir/build: libgp.so
.PHONY : CMakeFiles/libgp.so.dir/build

CMakeFiles/libgp.so.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/libgp.so.dir/cmake_clean.cmake
.PHONY : CMakeFiles/libgp.so.dir/clean

CMakeFiles/libgp.so.dir/depend:
	cd /scratch/yguo25/files/nnsys/deep-codegen/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /scratch/yguo25/files/nnsys/deep-codegen /scratch/yguo25/files/nnsys/deep-codegen /scratch/yguo25/files/nnsys/deep-codegen/build /scratch/yguo25/files/nnsys/deep-codegen/build /scratch/yguo25/files/nnsys/deep-codegen/build/CMakeFiles/libgp.so.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/libgp.so.dir/depend

