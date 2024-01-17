cmake_minimum_required(VERSION 3.12)

project(mesh_boolean_utils
    VERSION 0.1
    DESCRIPTION "Mesh boolean operation pybinding"
    LANGUAGES CXX)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build in Release mode by default
if (NOT CMAKE_BUILD_TYPE AND NOT MSVC)
  set(CMAKE_BUILD_TYPE "Release")
endif (NOT CMAKE_BUILD_TYPE AND NOT MSVC)

# Find the Python interpreter and libraries
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python executable: ${Python_EXECUTABLE}")

# Find pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG master
)
FetchContent_MakeAvailable(pybind11)

# Find libigl
include(libigl)
igl_include(copyleft cgal)


# Create the Python extension module
pybind11_add_module(mesh_boolean_utils atek/utils/mesh_boolean_utils.cpp)

# Include directories for libigl and pybind11
target_include_directories(mesh_boolean_utils PRIVATE ${pybind11_INCLUDE_DIR})
include_directories(atek/utils) # Path to your C++ header files

# Link Python, libigl, and any other necessary libraries
target_link_libraries(mesh_boolean_utils PRIVATE igl_copyleft::cgal)
