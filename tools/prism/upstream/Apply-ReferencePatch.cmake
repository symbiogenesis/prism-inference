# Optional explicit source substitutions; the verified upstream tree stays intact.
function(prism_reference_source variable relative_source before_sha256 after_sha256)
  if(NOT DEFINED ${variable})
    return()
  endif()
  if(NOT IS_ABSOLUTE "${${variable}}" OR NOT EXISTS "${${variable}}")
    message(FATAL_ERROR "${variable} must name the external verified source")
  endif()
  file(SHA256 "${PRISM_SOURCE_DIR}/${relative_source}" actual_before)
  file(SHA256 "${${variable}}" actual_after)
  if(NOT actual_before STREQUAL before_sha256 OR NOT actual_after STREQUAL after_sha256)
    message(FATAL_ERROR "Production source substitution differs from its explicit pin: ${relative_source}")
  endif()
  # Set source properties in the owning target's directory. All original source
  # files remain byte-identical to the archive and retain their cache identity.
  set_source_files_properties("${PRISM_SOURCE_DIR}/${relative_source}"
    TARGET_DIRECTORY llama PROPERTIES HEADER_FILE_ONLY TRUE)
  target_sources(llama PRIVATE "${${variable}}")
  target_include_directories(llama PRIVATE "${PRISM_SOURCE_DIR}/src")
endfunction()

prism_reference_source(PRISM_PATCHED_SAVER src/llama-model-saver.cpp
  6dec1fc297dca89fb6933b638acf593a2e4c1ca02b69dafe00268651fa9ef96d
  8b2deae2693ff2a664e3d53e82cea06f8166b4671eadd0ba4a957c2a5af53dee)
prism_reference_source(PRISM_PATCHED_VOCAB src/llama-vocab.cpp
  32b8b7ac4a023bc6367eb7832e56e988ab7373a676ddcca4977665f78932ab88
  ecb6ffd1b5248384b96360acbb47bd3a6b8f8aebb547dade3b9429d8c00ff6e2)
