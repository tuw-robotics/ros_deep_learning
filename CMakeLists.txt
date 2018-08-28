cmake_minimum_required(VERSION 2.8.3)
project(ros_deep_learning)

option(MONOCULAR_GP "compile node for monocular camera with ground plane dependency from rwth")

if(MONOCULAR_GP)
	find_package(catkin REQUIRED COMPONENTS
	  cv_bridge
	  image_transport
	  roscpp
	  sensor_msgs
	  std_msgs
	  tuw_object_msgs
	  rwth_perception_people_msgs
	)
else(MONOCULAR_GP)
	find_package(catkin REQUIRED COMPONENTS
	  cv_bridge
	  image_transport
	  roscpp
	  sensor_msgs
	  std_msgs
	  tuw_object_msgs
	)
endif(MONOCULAR_GP)

find_package(jetson-inference REQUIRED)
find_package(CUDA REQUIRED)
find_package(Qt4 REQUIRED)
find_package(Eigen3 REQUIRED)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

find_library(
	NVINFER_LIB
	NAMES libnvinfer nvinfer
	HINTS "/opt/TensorRT-4.0.1.6/lib"
)

message(STATUS "NVINFER_LIB: [${NVINFER_LIB}]")

find_library(
	NVCAFFE_PARSER_LIB
	NAMES libnvcaffe_parser nvcaffe_parser
	HINTS "/opt/TensorRT-4.0.1.6/lib"
)

message(STATUS "NVCAFFE_PARSER_LIB: [${NVCAFFE_PARSER_LIB}]")

catkin_package(
    LIBRARIES ros_deep_learning_nodelets
    #CATKIN_DEPENDS nodelet roscpp cv_bridge image_transport sensor_msgs tuw_object_msgs rwth_perception_people_msgs
)

# the library we are going to compile (has all of the nodelets in it)
if(MONOCULAR_GP)
	add_library(ros_deep_learning_nodelets src/ros_detectnet_monocular.cpp src/ros_detectnet_rgbd.cpp)
else(MONOCULAR_GP)
	add_library(ros_deep_learning_nodelets src/ros_detectnet_rgbd.cpp)
endif(MONOCULAR_GP)

# includes
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  /opt/TensorRT-4.0.1.6/include
)

add_dependencies(ros_deep_learning_nodelets ${catkin_EXPORTED_TARGETS})

# compile option
set_target_properties(ros_deep_learning_nodelets PROPERTIES COMPILE_FLAGS "-std=c++11")
# link agains these objects
target_link_libraries(ros_deep_learning_nodelets
  ${catkin_LIBRARIES}
  jetson-inference
)

#if(catkin_EXPORTED_LIBRARIES)
#    add_dependencies(ros_deep_learning_nodelets ${catkin_EXPORTED_LIBRARIES})
#endif()
