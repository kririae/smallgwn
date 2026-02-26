#pragma once

#include <cstdio>
#include <cstdlib>
#include <string_view>

#if defined(__CUDACC__) && defined(__NVCC__) && !defined(__CUDACC_RELAXED_CONSTEXPR__)
#error                                                                                             \
    "smallgwn requires NVCC flag --expt-relaxed-constexpr. Link gwn::smallgwn in CMake or add the flag manually."
#endif

namespace gwn::detail {

[[nodiscard]] constexpr std::string_view
gwn_assert_filename(std::string_view const path) noexcept {
    std::size_t const separator_index = path.find_last_of("/\\");
    if (separator_index == std::string_view::npos)
        return path;

    return path.substr(separator_index + 1);
}

[[noreturn]] inline void
gwn_assert_fail_host(char const *const condition, char const *const file, int const line) noexcept {
    std::fprintf(stderr, "%s:%d: Assertion (%s) failed.\n", file, line, condition);
    std::fflush(stderr);
    std::abort();
}

template <class... Args>
[[noreturn]] inline void gwn_assert_fail_host(
    char const *const condition, char const *const file, int const line,
    char const *const format, Args &&...args
) noexcept {
    std::fprintf(stderr, "%s:%d: Assertion (%s) failed. ", file, line, condition);
    std::fprintf(stderr, format, args...);
    std::fputc('\n', stderr);
    std::fflush(stderr);
    std::abort();
}

[[noreturn]] __device__ inline void
gwn_assert_fail_device(char const *const condition, char const *const file, int const line) noexcept {
    printf("%s:%d: Assertion (%s) failed.\n", file, line, condition);
    asm volatile("trap;");
    __builtin_unreachable();
}

template <class... Args>
[[noreturn]] __device__ inline void gwn_assert_fail_device(
    char const *const condition, char const *const file, int const line,
    char const *const format, Args &&...args
) noexcept {
    printf("%s:%d: Assertion (%s) failed. ", file, line, condition);
    printf(format, args...);
    printf("\n");
    asm volatile("trap;");
    __builtin_unreachable();
}

} // namespace gwn::detail

#if !defined(GWN_UNLIKELY)
#if defined(__clang__) || defined(__GNUC__)
#define GWN_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#else
#define GWN_UNLIKELY(condition) (!!(condition))
#endif
#endif

#if !defined(GWN_FILENAME)
#if defined(__CUDA_ARCH__)
#define GWN_FILENAME __FILE__
#else
#define GWN_FILENAME (::gwn::detail::gwn_assert_filename(std::string_view{__FILE__}).data())
#endif
#endif

#if !defined(GWN_DETAIL_ASSERT_FAIL)
#if defined(__CUDA_ARCH__)
#define GWN_DETAIL_ASSERT_FAIL(...) ::gwn::detail::gwn_assert_fail_device(__VA_ARGS__)
#else
#define GWN_DETAIL_ASSERT_FAIL(...) ::gwn::detail::gwn_assert_fail_host(__VA_ARGS__)
#endif
#endif

#if !defined(GWN_FORCE_ASSERT)
#define GWN_FORCE_ASSERT(condition, ...)                                                        \
    do {                                                                                        \
        if (GWN_UNLIKELY(!(condition)))                                                        \
            GWN_DETAIL_ASSERT_FAIL(#condition, GWN_FILENAME, __LINE__ __VA_OPT__(, )          \
                                   __VA_ARGS__);                                               \
    } while (false)
#endif

#if !defined(NDEBUG)
#if !defined(GWN_ASSERT)
#define GWN_ASSERT(condition, ...) GWN_FORCE_ASSERT(condition __VA_OPT__(, ) __VA_ARGS__)
#endif
#else
#if !defined(GWN_ASSERT)
#define GWN_ASSERT(condition, ...) ((void)sizeof(!(condition)))
#endif
#endif
