cmake_minimum_required(VERSION 3.0.0)
find_package(Python COMPONENTS Interpreter Development)
if (${Python_FOUND})
project(TensorCore_Python VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin)

include(CTest)
enable_testing()

add_library(tensor SHARED src/tensor_bind.cc)

target_include_directories(
    tensor
    PRIVATE "${CMAKE_CURRENT_LIST_DIR}/.venv/lib/python3.10/site-packages/pybind11/include"
    PRIVATE "/mnt/d/C++_ver2/TensorMain_wsl2/src"
    PRIVATE ${Python_INCLUDE_DIRS}
)

SET_TARGET_PROPERTIES(tensor PROPERTIES PREFIX "")

target_link_directories(
    tensor
    PRIVATE "/mnt/d/C++_ver2/TensorMain_wsl2/lib"
)

target_link_libraries(tensor TensorCore_Core)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

install(TARGETS tensor LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
endif()
