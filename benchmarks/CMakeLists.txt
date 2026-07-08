add_executable(smallgwn_benchmark benchmark_main.cu)
target_link_libraries(smallgwn_benchmark PRIVATE gwn::smallgwn)
target_include_directories(smallgwn_benchmark
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/tests"
)
smallgwn_apply_strict_warnings(smallgwn_benchmark)
