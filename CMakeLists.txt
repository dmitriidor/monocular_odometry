cmake_minimum_required(VERSION 3.0.0)
project(monovis_odom VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})

add_subdirectory(pose_estimator)

add_executable(main main.cpp)
target_link_libraries(main
    ${OpenCV_LIBS} 
    pose_estimator_class
    )
