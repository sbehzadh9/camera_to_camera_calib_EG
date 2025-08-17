cmake_minimum_required(VERSION 3.10)
project(FeatureMatchingEpipolar)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)

# If OpenCV was found, print the include dirs and link dirs (for debugging)
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(FeatureMatchingEpipolar main.cpp)

target_link_libraries(FeatureMatchingEpipolar ${OpenCV_LIBS})
