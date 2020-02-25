cmake_minimum_required(VERSION 3.8)

# compiler setting: use intel C/C++ compiler.
find_program(ICC icc)
if (ICC)
    message(STATUS "icc found: ${ICC}")
    set(CMAKE_C_COMPILER "${ICC}" CACHE FILEPATH "Intel C compiler" FORCE)
else()
    message(WARNING "icc not found.")
endif()
find_program(ICPC icpc)
if (ICPC)
    message(STATUS "icpc found: ${ICPC}")
    set(CMAKE_CXX_COMPILER "${ICPC}" CACHE FILEPATH "Intel C++ compiler" FORCE)
else()
    message(WARNING "icpc not found.")
endif()

enable_testing()

# C/C++ standard setting
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(project_name qe_dft)
project(${project_name} CXX)

# find eigen library
find_package (Eigen3 REQUIRED NO_MODULE)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 library found.")
else()
    message(FATAL_ERROR "Can not found library Eigen3.")
endif()

# blas library
find_library(
    BLAS_LIBRARY
    NAMES blas
    PATHS "/usr/lib")
if (BLAS_LIBRARY)
    message(STATUS "libblas: ${BLAS_LIBRARY}")
else()
    message(FATAL_ERROR "libblas not found")
endif()

# lapack library
find_library(
    LAPACK_LIBRARY
    NAMES lapack
    PATHS "/usr/lib")
if (BLAS_LIBRARY)
    message(STATUS "liblapack: ${LAPACK_LIBRARY}")
else()
    message(FATAL_ERROR "liblapack not found")
endif()

file(GLOB_RECURSE Sources
    "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_definitions(-DDEBUG_PRINT=off)
add_executable(
    ${project_name}
    ${Sources})

target_link_libraries(
    ${project_name}
    PUBLIC
    ${BLAS_LIBRARY}
    ${LAPACK_LIBRARY}
    Eigen3::Eigen
    /usr/local/lib/libboost_program_options.a
    /usr/local/lib/libboost_regex.a
    /usr/local/lib/libboost_filesystem.a)

if (OPENMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(
            ${project_name}
            PUBLIC
            OpenMP::OpenMP_CXX)
    endif()
endif()
