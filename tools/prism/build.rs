//! Generates the loader ABI from Rust and links opt-in pinned ggml tests.

use std::env;
use std::error::Error;
#[cfg(feature = "engine-loader")]
use std::fs;
#[cfg(feature = "engine-loader")]
use std::path::PathBuf;

#[cfg(feature = "engine-loader")]
fn generate_header() -> Result<(), Box<dyn Error>> {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or("missing manifest path")?);
    let source = manifest.join("src/loader.rs");
    let configuration = manifest.join("cbindgen.toml");
    println!("cargo::rerun-if-changed={}", source.display());
    println!("cargo::rerun-if-changed={}", configuration.display());
    let mut generator = cbindgen::Builder::new()
        .with_src(source)
        .with_config(cbindgen::Config::from_file(configuration)?);
    // cbindgen also parses the loader's imported ggml declarations. Only the
    // Rust-owned loader exports belong in the generated engine-facing ABI.
    let declarations = generator.clone().generate()?;
    for symbol in declarations
        .dynamic_symbols_names()
        .filter(|symbol| !symbol.starts_with("bonsai_prism_loader_"))
    {
        generator = generator.exclude_item(symbol);
    }
    let bindings = generator.generate()?;
    let mut exports: Vec<_> = bindings.dynamic_symbols_names().collect();
    exports.sort_unstable();
    if exports
        != [
            "bonsai_prism_loader_error",
            "bonsai_prism_loader_free",
            "bonsai_prism_loader_load",
            "bonsai_prism_loader_make",
            "bonsai_prism_loader_new",
            "bonsai_prism_loader_shared_bytes",
            "bonsai_prism_loader_shared_tensors",
        ]
    {
        return Err(
            format!("generated loader ABI differs from the public exports: {exports:?}").into(),
        );
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").ok_or("missing build output path")?);
    let mut header = Vec::new();
    bindings.write(&mut header);
    fs::write(output.join("bonsai_prism.h"), header)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "engine-loader")]
    generate_header()?;
    println!("cargo::rerun-if-env-changed=BONSAI_PRISM_NATIVE_LIB_DIR");
    println!("cargo::rerun-if-env-changed=BONSAI_PRISM_NATIVE_LINK_KIND");
    if env::var_os("CARGO_FEATURE_NATIVE_TESTS").is_none() {
        return Ok(());
    }
    let directories = env::var_os("BONSAI_PRISM_NATIVE_LIB_DIR").ok_or(
        "native-tests requires BONSAI_PRISM_NATIVE_LIB_DIR with absolute ggml library directories",
    )?;
    let mut found_directory = false;
    for directory in env::split_paths(&directories) {
        if !directory.is_absolute()
            || !directory.is_dir()
            || directory.to_string_lossy().contains('%')
        {
            return Err("BONSAI_PRISM_NATIVE_LIB_DIR entries must be existing, expanded absolute directories".into());
        }
        println!("cargo::rustc-link-search=native={}", directory.display());
        found_directory = true;
    }
    if !found_directory {
        return Err("BONSAI_PRISM_NATIVE_LIB_DIR must contain a directory".into());
    }
    let kind = env::var("BONSAI_PRISM_NATIVE_LINK_KIND").unwrap_or_else(|_| "static".to_owned());
    if kind != "static" && kind != "dylib" {
        return Err("BONSAI_PRISM_NATIVE_LINK_KIND must be static or dylib".into());
    }
    for library in ["ggml", "ggml-cpu", "ggml-base"] {
        println!("cargo::rustc-link-lib={kind}={library}");
    }
    if kind == "static" {
        match env::var("CARGO_CFG_TARGET_OS")?.as_str() {
            "linux" => {
                for library in ["stdc++", "m", "dl", "pthread"] {
                    println!("cargo::rustc-link-lib={library}");
                }
            }
            "macos" => println!("cargo::rustc-link-lib=c++"),
            // MSVC-built ggml carries its runtime's default-library directives.
            "windows" => {}
            other => {
                return Err(format!(
                    "native-tests does not configure ggml runtime libraries for {other}"
                )
                .into());
            }
        }
    }
    Ok(())
}
