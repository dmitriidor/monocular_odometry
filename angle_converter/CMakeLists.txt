add_library(
    angle_converter_class
    angle_converter_class.cpp
    angle_converter_class.hpp
)

find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})
target_link_libraries(angle_converter_class ${Eigen3_LIBS})

target_compile_definitions(angle_converter_class PUBLIC ANGLE_CONVERTER_VERSION=1)