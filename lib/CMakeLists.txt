aux_source_directory("../src" SRC_LIST)
SET(LIBCALIBRATION_SRC ${SRC_LIST})
ADD_LIBRARY(caculateRadiusAndDistance STATIC ${LIBCALIBRATION_SRC})

include_directories( "../include")

#opencv
find_package(OpenCV REQUIRED)
include_directories(
${OpenCV_INCLUDE_DIRS}
)

#ceres
find_package(Ceres REQUIRED)
include_directories( ${CERES_INCLUDE_DIRS})

#eigen
include_directories( "/usr/include/eigen3/")

#pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


