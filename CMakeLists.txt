cmake_minimum_required(VERSION 3.0.2)
project(stereo-tuner)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(OpenCV 4.5.0 REQUIRED)
find_package(OpenCV 4.5.0 REQUIRED)
find_package(OpenCV 4.5.0 REQUIRED)
find_package ( PkgConfig REQUIRED )
PKG_CHECK_MODULES( GTK REQUIRED gtk+-3.0 )

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
${OpenCV_INCLUDE_DIRS}
${GTK_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(${PROJECT_NAME} main.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${GTK_LIBRARIES}
)
