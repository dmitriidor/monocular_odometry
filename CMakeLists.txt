cmake_minimum_required(VERSION 3.0.0)
project(monovis_odom VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)

# set(OpenCV_DIR "/home/dmitrii/Documents/opencv_build/installation_opencv/include/opencv4/opencv2")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

add_subdirectory(pose_estimator)
add_subdirectory(tile_depthmeter)
add_subdirectory(test_vids)
add_subdirectory(angle_converter)

add_executable(monovis_odom main.cpp)
target_link_libraries(monovis_odom
    ${OpenCV_LIBS} 
    pose_estimator_class
    angle_converter_class
    tile_depthmeter_class
    )
