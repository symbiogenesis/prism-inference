# Build only the Rust process adapter. ggml itself remains the pinned native
# production library; the Rust conformance harness owns all test semantics.
function(add_prism_rust_ggml_bridge)
  find_program(PRISM_RUSTUP_EXECUTABLE rustup REQUIRED)
  set(PRISM_RUST_TOOLCHAIN "nightly-2026-10-22" CACHE STRING "Pinned Rust toolchain for the ggml process adapter")
  if(NOT PRISM_RUST_TOOLCHAIN MATCHES "^[A-Za-z0-9_.-]+$")
    message(FATAL_ERROR "PRISM_RUST_TOOLCHAIN must be one explicit rustup toolchain name")
  endif()

  # CMake's generator platform can differ from the machine running rustup.
  # In particular, native Windows ARM64 hosts also build x64 reference tools.
  if(WIN32)
    if(CMAKE_GENERATOR_PLATFORM MATCHES "^[Aa][Rr][Mm]64" OR
        (NOT CMAKE_GENERATOR_PLATFORM AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$"))
      set(rust_target aarch64-pc-windows-msvc)
    elseif(CMAKE_GENERATOR_PLATFORM MATCHES "^[Xx]64" OR
        (NOT CMAKE_GENERATOR_PLATFORM AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$"))
      set(rust_target x86_64-pc-windows-msvc)
    else()
      message(FATAL_ERROR "The Rust ggml adapter supports Windows ARM64 and x64")
    endif()
    # The pinned CPU backend queries the Windows processor registry.
    set(native_runtime_args -l dylib=advapi32)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
      set(rust_target aarch64-unknown-linux-gnu)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
      set(rust_target x86_64-unknown-linux-gnu)
    else()
      message(FATAL_ERROR "The Rust ggml adapter supports Linux ARM64 and x64")
    endif()
    set(native_runtime_args -l dylib=stdc++ -l dylib=m -l dylib=dl -l dylib=pthread)
  else()
    message(FATAL_ERROR "The Rust ggml adapter supports Windows and Linux reference hosts")
  endif()

  if(CMAKE_CONFIGURATION_TYPES)
    set(output_directory "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
  else()
    set(output_directory "${CMAKE_CURRENT_BINARY_DIR}")
  endif()
  set(output "${output_directory}/prism-upstream-bridge${CMAKE_EXECUTABLE_SUFFIX}")
  set(source_directory "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/bridge")
  add_custom_command(OUTPUT "${output}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_directory}"
    COMMAND "${PRISM_RUSTUP_EXECUTABLE}" run "${PRISM_RUST_TOOLCHAIN}" rustfmt
      --edition 2024 --check "${source_directory}/main.rs"
    COMMAND "${PRISM_RUSTUP_EXECUTABLE}" run "${PRISM_RUST_TOOLCHAIN}" clippy-driver
      --edition 2024 --crate-name prism_upstream_bridge --target "${rust_target}"
      --emit metadata -D warnings -W clippy::all -W clippy::pedantic -W clippy::nursery
      "${source_directory}/main.rs" -o "${output_directory}/bridge.rmeta"
    COMMAND "${PRISM_RUSTUP_EXECUTABLE}" run "${PRISM_RUST_TOOLCHAIN}" rustc
      "${source_directory}/main.rs" --crate-name prism_upstream_bridge
      --edition 2024 --crate-type bin --target "${rust_target}"
      -D warnings -C opt-level=3 -C overflow-checks=yes
      "-Lnative=$<TARGET_FILE_DIR:ggml>"
      "-Lnative=$<TARGET_FILE_DIR:ggml-cpu>"
      "-Lnative=$<TARGET_FILE_DIR:ggml-base>"
      -l static=ggml -l static=ggml-cpu -l static=ggml-base
      ${native_runtime_args} -o "${output}"
    DEPENDS "${source_directory}/main.rs" "${source_directory}/protocol.rs"
      "${source_directory}/native.rs" "${CMAKE_CURRENT_FUNCTION_LIST_FILE}" ggml ggml-cpu ggml-base
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building the Rust production ggml process adapter (${rust_target})"
    VERBATIM COMMAND_EXPAND_LISTS)
  add_custom_target(prism-upstream-bridge ALL DEPENDS "${output}")
endfunction()
