cmake_minimum_required(VERSION 3.24)

project(GMM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED)

add_subdirectory(src)

#find_package(LibArchive REQUIRED)

add_executable(main)
target_sources( main main.cpp
              )
target_include_directories(main 
                          PUBLIC "${CMAKE_SOURCE_DIR}/include" 
)

### Unit Tests (TBD)
#add_subdirectory(test)

