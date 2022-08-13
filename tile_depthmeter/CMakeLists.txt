add_library(
    tile_depthmeter_class
    tile_depthmeter_class.cpp
    tile_depthmeter_class.hpp
)

target_compile_definitions(tile_depthmeter_class PUBLIC TILE_DEPTHMETER_VERSION=1)