cmake_minimum_required(VERSION 3.24)
project(BonsaiPrismUpstreamCommonAdapter LANGUAGES C CXX)

if(NOT DEFINED PRISM_SOURCE_DIR)
  message(FATAL_ERROR "PRISM_SOURCE_DIR must name the verified, unmodified pinned Prism source")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
foreach(option BUILD_TESTING LLAMA_BUILD_TESTS LLAMA_BUILD_TOOLS LLAMA_BUILD_SERVER
    LLAMA_BUILD_APP LLAMA_BUILD_EXAMPLES LLAMA_BUILD_UI MTMD_VIDEO
    LLAMA_USE_PREBUILT_UI LLAMA_OPENSSL LLAMA_LLGUIDANCE LLAMA_SUBPROCESS
    LLAMA_DSPARK_MARKOV_BLAS LLAMA_DSPARK_MARKOV_CUDA LLAMA_DSPARK_MARKOV_METAL
    GGML_CCACHE GGML_BACKEND_DL GGML_NATIVE GGML_CPU_ALL_VARIANTS GGML_OPENMP
    GGML_CUDA GGML_METAL GGML_ACCELERATE GGML_BLAS GGML_LLAMAFILE)
  set(${option} OFF CACHE BOOL "" FORCE)
endforeach()
# Configure only vendored dependencies; an adapter build must not fetch packages.
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "" FORCE)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "" FORCE)
add_subdirectory("${PRISM_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/prism")
include("${CMAKE_CURRENT_LIST_DIR}/../Apply-ReferencePatch.cmake")

# Consume a separately pinned/verified production LLGuidance build. This enables
# the pristine common/llguidance.cpp feature branch without network build steps.
if(DEFINED PRISM_LLGUIDANCE_LIBRARY OR DEFINED PRISM_LLGUIDANCE_INCLUDE_DIR)
  if(NOT IS_ABSOLUTE "${PRISM_LLGUIDANCE_LIBRARY}" OR
      NOT EXISTS "${PRISM_LLGUIDANCE_LIBRARY}" OR
      NOT IS_ABSOLUTE "${PRISM_LLGUIDANCE_INCLUDE_DIR}" OR
      NOT EXISTS "${PRISM_LLGUIDANCE_INCLUDE_DIR}/llguidance.h")
    message(FATAL_ERROR "Supply both absolute verified LLGuidance library and header paths")
  endif()
  add_library(prism-llguidance STATIC IMPORTED)
  set_target_properties(prism-llguidance PROPERTIES IMPORTED_LOCATION "${PRISM_LLGUIDANCE_LIBRARY}")
  target_compile_definitions(llama-common PUBLIC LLAMA_USE_LLGUIDANCE)
  target_include_directories(llama-common PRIVATE "${PRISM_LLGUIDANCE_INCLUDE_DIR}")
  target_link_libraries(llama-common PRIVATE prism-llguidance)
  if(WIN32)
    target_link_libraries(llama-common PRIVATE ws2_32 userenv ntdll bcrypt synchronization)
  endif()
endif()

# Only these two production units supply the chat conversion APIs used here.
# Do not compile the server context, task, queue, transport or MCP machinery.
find_package(Threads REQUIRED)
set(PRISM_SERVER_CONTEXT_SOURCES
  server-chat.cpp server-common.cpp)
list(TRANSFORM PRISM_SERVER_CONTEXT_SOURCES PREPEND "${PRISM_SOURCE_DIR}/tools/server/")
add_library(prism-upstream-server-context STATIC ${PRISM_SERVER_CONTEXT_SOURCES})
target_include_directories(prism-upstream-server-context PRIVATE
  "${PRISM_SOURCE_DIR}/tools/server" "${PRISM_SOURCE_DIR}/tools/mtmd"
  "${PRISM_SOURCE_DIR}/src" "${PRISM_SOURCE_DIR}")
target_link_libraries(prism-upstream-server-context PUBLIC llama-common mtmd Threads::Threads)
if(MSVC)
  target_compile_options(prism-upstream-server-context PRIVATE /utf-8)
endif()

add_executable(prism-upstream-common-bridge bridge.cpp action_file.hpp jinja.hpp mtmd.hpp grammar_parser.hpp reasoning_budget.hpp tensor_graph.hpp grammar.hpp codec.hpp peg.hpp concurrency.hpp arena_graph.hpp batch_allocator.hpp allocation.hpp mtmd_chunks.hpp sampling.hpp gguf.hpp chat_templates.hpp chat_parser.hpp chat_auto.hpp vocab.hpp vocab_session.hpp model_runtime.hpp json_schema.hpp chat_analysis.hpp public_c_abi.c public_c_abi.h arg_parser.hpp model_resolution.hpp chat.hpp generated_model.hpp optimizer.hpp quantization.hpp dspark.hpp threadpool.hpp backend_ops.hpp backend_session.hpp dspark_runtime.hpp llguidance.hpp backend_sampler.hpp context_graph.hpp codec_buffers.hpp)
target_include_directories(prism-upstream-common-bridge PRIVATE "${PRISM_SOURCE_DIR}/src" "${PRISM_SOURCE_DIR}" "${PRISM_SOURCE_DIR}/ggml/src")
target_link_libraries(prism-upstream-common-bridge PRIVATE llama-common mtmd prism-upstream-server-context)
# The same-revision CI tool bundle also supplies the small ggml-only wrapper.
# Reuse this tree's production ggml libraries instead of compiling them again in
# a second CMake project. Ordinary common-only builds never request this target.
add_executable(prism-upstream-bridge ../bridge.cpp)
target_link_libraries(prism-upstream-bridge PRIVATE ggml ggml-cpu)
# The pinned public C ABI intentionally throws for source-tested invalid inputs.
# Preserve those C++ exception observations instead of assuming extern C is no-throw.
if(MSVC)
  set_property(SOURCE bridge.cpp APPEND PROPERTY COMPILE_OPTIONS /EHs /EHc-)
endif()

set_source_files_properties(public_c_abi.c PROPERTIES LANGUAGE C)
set_target_properties(prism-upstream-common-bridge PROPERTIES C_STANDARD 17 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)

# Actual production LoRA exporter; no upstream test translation unit.
add_executable(llama-export-lora "${PRISM_SOURCE_DIR}/tools/export-lora/export-lora.cpp")
target_link_libraries(llama-export-lora PRIVATE llama-common llama)
