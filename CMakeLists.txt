
cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(simple_cuda LANGUAGES CXX CUDA)

# Enable better clangd support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build the library
# add_library(SimpleCudaLib STATIC
#   src/include/utils.h
# )
# set_target_properties(SimpleCudaLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_compile_features(SimpleCudaLib PUBLIC cxx_std_20)

# # Add include directories to the library
# target_include_directories(SimpleCudaLib PUBLIC src/include)


# Set the C++ standard for all targets
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Flags
set(EXTRA_CUDA_FLAGS "")

# Set the output directory for the binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Build the examples
file(GLOB EXAMPLE_SOURCES examples/*.cu)
foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})

    # CUDA properties provided by CMAKE
    set_target_properties(${EXAMPLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES CUDA_ARCHITECTURES 90)

    # Add  Extra CUDA flags
    target_compile_options(${EXAMPLE_NAME} PRIVATE ${EXTRA_CUDA_FLAGS})

    # Include header utils
    # This is gross but makes clangd work, I am sure is brittle but w.e for now
    target_include_directories(${EXAMPLE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    # Link the library to the target
    # target_link_libraries(${EXAMPLE_NAME} SimpleCudaLib)
endforeach()