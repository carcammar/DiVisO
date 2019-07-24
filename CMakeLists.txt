cmake_minimum_required(VERSION 2.8.12)
project(DiVisO)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

# Check c++11 (Required at least for ceres)
include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX17)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   add_definitions(-DCOMPILEDWITHC17)
   message(STATUS "Using flag -std=c++17.")
elseif(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


find_package(OpenCV 3.0 QUIET)
find_package(Eigen3 3.1.0 REQUIRED)
# find_package(Ceres QUIET)
find_package(Pangolin REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/lib
${EIGEN3_INCLUDE_DIR}
# ${CERES_INCLUDE_DIRS}
${Pangolin_INCLUDE_DIRS}
)

set(SOURCES
    launcher.cpp
    src/slam.cpp
    src/frame.cpp
    src/point.cpp
    src/camera.cpp
    src/maths.cpp
    src/display.cpp
)

set(HEADERS
    lib/slam.h
    lib/frame.h
    lib/point.h
    lib/camera.h
    lib/maths.h
    lib/display.h
)


add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${HEADERS})


target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
# ${CERES_LIBRARIES}
${Pangolin_LIBRARIES}
)

