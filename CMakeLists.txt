cmake_minimum_required(VERSION 3.18.0)

project(TensorArray_Python VERSION 0.1.0 LANGUAGES C CXX CUDA)

include(CTest)
enable_testing()

add_subdirectory("third_party/pybind11" EXCLUDE_FROM_ALL)
add_subdirectory("tensor-array-repo/Tensor-Array" EXCLUDE_FROM_ALL)
pybind11_add_module(tensor2 SHARED src/tensor_array/_core/tensor_bind.cc)

target_include_directories(
    tensor2
    PUBLIC "${CMAKE_CURRENT_LIST_DIR}/tensor-array-repo/Tensor-Array/src"
    INTERFACE ${Python_INCLUDE_DIRS}
)

SET_TARGET_PROPERTIES(tensor2 PROPERTIES PREFIX "")

target_link_libraries(tensor2 PUBLIC TensorArray::Core)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
