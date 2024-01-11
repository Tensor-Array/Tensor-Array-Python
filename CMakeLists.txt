cmake_minimum_required(VERSION 3.18.0)
find_package(Python COMPONENTS Interpreter Development)
if (${Python_FOUND})

project(TensorArray_Python VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/src/tensor_array/core)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin)

include(CTest)
enable_testing()

add_subdirectory("third_party/pybind11" EXCLUDE_FROM_ALL)
add_subdirectory("tensor-array-repo/Tensor-Array" EXCLUDE_FROM_ALL)
add_library(tensor2 SHARED src/tensor_array/_core/tensor_bind.cc)

target_include_directories(
    tensor2
    PRIVATE "${CMAKE_CURRENT_LIST_DIR}/third_party/pybind11/include"
    PRIVATE "${CMAKE_CURRENT_LIST_DIR}/tensor-array-repo/Tensor-Array/src"
    PRIVATE ${Python_INCLUDE_DIRS}
)

SET_TARGET_PROPERTIES(tensor2 PROPERTIES PREFIX "")

target_link_libraries(tensor2 PRIVATE TensorArray::Core)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

install(TARGETS tensor2 LIBRARY DESTINATION /src/tensor_array/core ARCHIVE DESTINATION lib)
endif()

