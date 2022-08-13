add_library(
    pose_estimator_class
    pose_estimator_class.cpp
    pose_estimator_class.hpp
)

find_package(OpenCV REQUIRED)
target_link_libraries(pose_estimator_class ${OpenCV_LIBS})

target_compile_definitions(pose_estimator_class PUBLIC POSE_ESTIMATOR_VERSION=1)