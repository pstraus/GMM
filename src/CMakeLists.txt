find_package(Eigen3 CONFIG REQUIRED)

add_library(GMPHD GMPHD.cpp)
target_link_libraries( GMPHD PUBLIC Eigen3::Eigen)
target_include_directories(GMPHD PUBLIC "${CMAKE_SOURCE_DIR}/include")

add_executable(main)
target_sources( main PRIVATE "main.cpp")

TARGET_LINK_LIBRARIES(main
											PUBLIC GMPHD)
              
target_include_directories(main 
                          PUBLIC "${CMAKE_SOURCE_DIR}/include" 
)