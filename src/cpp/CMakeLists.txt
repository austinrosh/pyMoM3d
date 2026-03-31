cmake_minimum_required(VERSION 3.18)
project(pyMoM3d_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Build type defaults
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
  else()
    add_compile_options(-O3 -march=native -DNDEBUG)
  endif()
endif()

# ---------------------------------------------------------------------------
# pybind11 (required)
# ---------------------------------------------------------------------------
find_package(pybind11 REQUIRED CONFIG
  HINTS "${CMAKE_SOURCE_DIR}/../../venv/lib/python3.13/site-packages/pybind11/share/cmake/pybind11"
)

# ---------------------------------------------------------------------------
# NumPy include directory (required for array_t<>)
# ---------------------------------------------------------------------------
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${NUMPY_INCLUDE_DIR})

# ---------------------------------------------------------------------------
# OpenMP (optional — falls back to single-threaded if not found)
# ---------------------------------------------------------------------------
find_package(OpenMP OPTIONAL_COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found — parallel build enabled")
else()
  message(STATUS "OpenMP NOT found — building single-threaded")
endif()

# macOS + Apple clang: point to libomp from Homebrew
if(APPLE AND NOT OpenMP_CXX_FOUND)
  execute_process(
    COMMAND brew --prefix libomp
    OUTPUT_VARIABLE LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(LIBOMP_PREFIX)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor;-fopenmp;-I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    find_package(OpenMP COMPONENTS CXX)
  endif()
endif()

# ---------------------------------------------------------------------------
# Extension module
# ---------------------------------------------------------------------------
pybind11_add_module(_cpp_kernels
  mom_kernel.cpp
)

target_include_directories(_cpp_kernels PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}    # picks up singularity.hpp
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(_cpp_kernels PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install into the pyMoM3d.mom Python package
install(
  TARGETS _cpp_kernels
  DESTINATION "${CMAKE_SOURCE_DIR}/../../src/pyMoM3d/mom"
)
