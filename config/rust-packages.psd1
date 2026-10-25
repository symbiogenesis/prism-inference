@{
    Packages = @(
        @{ Name = 'bonsai-ternary'; Manifest = 'tools/ternary/Cargo.toml'; Group = 'Runtime'; Features = @('*') }
        @{ Name = 'bonsai-gguf'; Manifest = 'tools/gguf/Cargo.toml'; Group = 'Runtime'; Features = @('*') }
        @{ Name = 'bonsai-prism'; Manifest = 'tools/prism/Cargo.toml'; Group = 'Runtime'; Features = @(); CompatibilityCompileFeatures = @('engine-loader'); NativeFeatures = @('*') }
        @{ Name = 'prism-inference'; Manifest = 'Cargo.toml'; Group = 'Runtime'; Features = @('*') }
        @{ Name = 'prism-inference-server'; Manifest = 'server/Cargo.toml'; Group = 'Runtime'; Features = @('*') }
    )
}