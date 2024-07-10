cmake_minimum_required(VERSION 3.15...3.29)
project(PyLRI LANGUAGES CXX)

set(pybind11_DIR ./pybind11/tools/)
set(PYBIND11_FINDPYTHON ON)
# find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(pybind11)

pybind11_add_module(PyLRI global/Tensor.cpp)
install(TARGETS PyLRI DESTINATION .)