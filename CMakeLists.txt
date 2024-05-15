
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(simple_cuda LANGUAGES CXX CUDA)
# Set the C++ standard for all targets
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable better clangd support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#  simple_cuda source files
file(GLOB_RECURSE SIMPLE_CUDA_CU_SOURCES src/*.cu)
file(GLOB_RECURSE SIMPLE_CUDA_CPP_SOURCES src/*.cpp)
MESSAGE(STATUS "SIMPLE_CUDA_CU_SOURCES: ${SIMPLE_CUDA_CU_SOURCES}")
MESSAGE(STATUS "SIMPLE_CUDA_CPP_SOURCES: ${SIMPLE_CUDA_CPP_SOURCES}")

add_library(SimpleCudaLib STATIC
  ${SIMPLE_CUDA_CU_SOURCES}
  ${SIMPLE_CUDA_CPP_SOURCES}
)

set_target_properties(SimpleCudaLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add include directories to the library
target_include_directories(SimpleCudaLib PUBLIC src/include)

# Add fmtlib
add_subdirectory(third_party/fmt)

# CUDA Flags
set(EXTRA_CUDA_FLAGS "${EXTRA_CUDA_FLAGS} -lineinfo")
option (SHOW_PTXAS_INFO "Show ptxas info" OFF)
if(SHOW_PTXAS_INFO)
    set(EXTRA_CUDA_FLAGS "${EXTRA_CUDA_FLAGS} -Xptxas -v")
endif()
# Set the output directory for the binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Build the examples
file(GLOB_RECURSE EXAMPLE_SOURCES examples/*.cu)
foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})

    # CUDA properties provided by CMAKE
    set_target_properties(${EXAMPLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES CUDA_ARCHITECTURES 90a)

    # Convert the flags string into a list of flags
    separate_arguments(EXTRA_CUDA_FLAGS_LIST UNIX_COMMAND "${EXTRA_CUDA_FLAGS}")

    # Add Extra CUDA flags
    target_compile_options(${EXAMPLE_NAME} PRIVATE ${EXTRA_CUDA_FLAGS_LIST})

    # Include header utils
    # This is gross but makes clangd work, I am sure is brittle but w.e for now
    target_include_directories(${EXAMPLE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    # Third party links
    target_link_libraries(${EXAMPLE_NAME} PRIVATE fmt::fmt)

    # Link the library to the target
    target_link_libraries(${EXAMPLE_NAME} PRIVATE SimpleCudaLib)

endforeach()
