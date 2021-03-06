# CMake 最低版本号要求
cmake_minimum_required(VERSION 3.0.0)
# 项目信息
project(CaculateRadiusAndDistance VERSION 0.1 LANGUAGES CXX)

ADD_SUBDIRECTORY(lib)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_BUILD_TYPE "Release")

#opencv
find_package(OpenCV REQUIRED)
include_directories(
${OpenCV_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR}/include
)

#ceres
find_package(Ceres REQUIRED)
include_directories( ${CERES_INCLUDE_DIRS})

#eigen
#find_package(Eigen REQUIRED)
include_directories( "/usr/include/eigen3/")

#pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

aux_source_directory(src SRC_LIST)
aux_source_directory(demo DEMO_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST} ${DEMO_LIST})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${PCL_LIBRARIES})




