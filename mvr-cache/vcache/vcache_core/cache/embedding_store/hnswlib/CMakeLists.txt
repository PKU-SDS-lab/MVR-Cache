cmake_minimum_required(VERSION 3.0...3.26)

project(hnswlib
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hnswlib/third/Eigen>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS hnswlib
    EXPORT hnswlibTargets)

install(EXPORT hnswlibTargets
    FILE hnswlibConfig.cmake
    NAMESPACE hnswlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib)

# Examples and tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()
if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 11)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
      check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
      if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
        message("set -march=native flag")
      else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
          SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
          message("set -mcpu=apple-m1 flag")
        endif()
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
    endif()



    add_executable(singlevector_search tests/cpp/singlevector_search.cpp)
    target_link_libraries(singlevector_search hnswlib)

    add_executable(multivector_search tests/cpp/multivector_search.cpp)
    target_link_libraries(multivector_search hnswlib)

    add_executable(multivector_search_efficient tests/cpp/multivector_search_efficient.cpp)
    target_link_libraries(multivector_search_efficient hnswlib)

    add_executable(multivector_search_efficientpost tests/cpp/multivector_search_efficientpost.cpp)
    target_link_libraries(multivector_search_efficientpost hnswlib)

    add_executable(multivector_search_skippingnodes tests/cpp/multivector_search_skippingnodes.cpp)
    target_link_libraries(multivector_search_skippingnodes hnswlib)

    add_executable(multivector_search_skippingnodes_query tests/cpp/multivector_search_skippingnodes_query.cpp)
    target_link_libraries(multivector_search_skippingnodes_query hnswlib)

    add_executable(multivector_search_efficientpost_approx tests/cpp/multivector_search_efficientpost_approx.cpp)
    target_link_libraries(multivector_search_efficientpost_approx hnswlib)

    add_executable(multivector_search_efficientpost_approx_dynamic tests/cpp/multivector_search_efficientpost_approx_dynamic.cpp)
    target_link_libraries(multivector_search_efficientpost_approx_dynamic hnswlib)


    add_executable(multivector_search_nihcc tests/cpp/multivector_search_nihcc.cpp)
    target_link_libraries(multivector_search_nihcc hnswlib)
    
    add_executable(test tests/cpp/test.cpp)
    target_link_libraries(test hnswlib)

    add_executable(test_cluster tests/cpp/test_cluster.cpp)
    target_link_libraries(test_cluster hnswlib)

    add_executable(raw tests/cpp/raw.cpp)
    target_link_libraries(raw hnswlib)

    add_executable(raw_nihcc tests/cpp/raw_nihcc.cpp)
    target_link_libraries(raw_nihcc hnswlib)

    add_executable(raw_single tests/cpp/raw_single.cpp)
    target_link_libraries(raw_single hnswlib)

    add_executable(multivector_search_parallel_inner tests/cpp/multivector_search_parallel_inner.cpp)
    target_link_libraries(multivector_search_parallel_inner hnswlib)

    add_executable(multivector_search_parallel_inner_k tests/cpp/multivector_search_parallel_inner_k.cpp)
    target_link_libraries(multivector_search_parallel_inner_k hnswlib)


    add_executable(multivector_search_self tests/cpp/multivector_search_self.cpp)
    target_link_libraries(multivector_search_self hnswlib)
endif()
