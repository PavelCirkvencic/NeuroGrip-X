cmake_minimum_required(VERSION 3.8)
project(neurogrip_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VehicleState.msg"
  "msg/DynamicsModel.msg"
  "msg/SafetyStatus.msg"
  DEPENDENCIES std_msgs
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(message_serialization test/test_message_serialization.py
    APPEND_ENV PYTHONPATH="${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_py")
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
