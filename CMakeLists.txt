cmake_minimum_required(VERSION 2.4.6)

project(hand_tracking_ros_package)

find_package(catkin REQUIRED message_generation geometry_msgs sensor_msgs rospy)

# add_message_files(
#   FILES
#   FaceKeyframeRequest.msg
#   FaceRequest.msg
#   Keyframe.msg
#   LookatRequest.msg
# )

generate_messages(DEPENDENCIES geometry_msgs sensor_msgs)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY scripts
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)