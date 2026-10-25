// Bounded byte transport only; callers validate paths and JSON action schemas.
#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace prism_action_file {
constexpr std::size_t max_bytes = 128ULL * 1024 * 1024;

inline std::string read(const std::filesystem::path & filename, const char * description) {
    const auto check = [description](bool valid, const char * reason) {
        if (!valid) throw std::runtime_error(std::string(description) + " " + reason);
    };
    check(std::filesystem::is_regular_file(filename), "must be regular");
#if defined(_WIN32)
    // Atomic publication can leave a short-lived delete-access handle. The CRT
    // ifstream sharing mode omits FILE_SHARE_DELETE and can reject valid bytes.
    struct owned_file {
        HANDLE value;
        ~owned_file() { if (value != INVALID_HANDLE_VALUE) CloseHandle(value); }
    };
    const auto io_error = [description](const char * operation) {
        const auto error = GetLastError();
        return std::system_error(static_cast<int>(error), std::system_category(),
                                 std::string(description) + " " + operation);
    };
    owned_file input{CreateFileW(filename.c_str(), GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr)};
    if (input.value == INVALID_HANDLE_VALUE) throw io_error("open");
    LARGE_INTEGER size;
    if (!GetFileSizeEx(input.value, &size)) throw io_error("size");
    check(size.QuadPart >= 0 && static_cast<std::uint64_t>(size.QuadPart) <= max_bytes,
          "exceeds bound");
    std::string bytes(static_cast<std::size_t>(size.QuadPart), '\0');
    DWORD count = 0;
    if (!ReadFile(input.value, bytes.data(), static_cast<DWORD>(bytes.size()), &count, nullptr))
        throw io_error("read");
    check(count == bytes.size(), "read extent differs");
    char extra;
    if (!ReadFile(input.value, &extra, 1, &count, nullptr)) throw io_error("EOF read");
    check(count == 0, "size changed");
    LARGE_INTEGER final_size;
    if (!GetFileSizeEx(input.value, &final_size)) throw io_error("final size");
    check(final_size.QuadPart == size.QuadPart, "size changed");
    return bytes;
#else
    const auto size = std::filesystem::file_size(filename);
    check(size <= max_bytes, "exceeds bound");
    std::ifstream input(filename, std::ios::binary);
    check(input.is_open(), "open failed");
    std::string bytes(static_cast<std::size_t>(size), '\0');
    if (size) input.read(bytes.data(), static_cast<std::streamsize>(size));
    check(bool(input), "read extent differs");
    check(input.get() == std::char_traits<char>::eof() && input.eof() && !input.bad(),
          "size changed");
    input.clear();
    input.seekg(0, std::ios::end);
    check(bool(input) && input.tellg() == static_cast<std::streamoff>(size), "size changed");
    return bytes;
#endif
}
} // namespace prism_action_file
