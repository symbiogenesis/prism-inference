@{
    Packages = @(
        @{ Name = 'bonsai-native-hash'; Manifest = 'tools/native-hash/Cargo.toml'; Group = 'Runtime'; Features = @('*') }
        @{ Name = 'bonsai-rust-trial'; Manifest = 'tools/rust-trial/Cargo.toml'; Group = 'Runtime'; Features = @('*') }
        @{ Name = 'bonsai-hexagon-native'; Manifest = 'tools/hexagon-native/Cargo.toml'; Group = 'Runtime'; Features = @('*') }
        @{ Name = 'bonsai-ternary'; Manifest = 'tools/ternary/Cargo.toml'; Group = 'Runtime'; Features = @('*') }
        @{ Name = 'bonsai-gguf'; Manifest = 'tools/gguf/Cargo.toml'; Group = 'Runtime'; Features = @('*') }
        @{ Name = 'bonsai-prism'; Manifest = 'tools/prism/Cargo.toml'; Group = 'Runtime'; Features = @(); CompatibilityCompileFeatures = @('engine-loader'); NativeFeatures = @('*') }
        @{ Name = 'prism-inference'; Manifest = 'Cargo.toml'; Group = 'Runtime'; Features = @('*') }
        @{ Name = 'prism-conformance-policy'; Manifest = 'tools/conformance-policy/Cargo.toml'; Group = 'Runtime'; Features = @('*') }
        @{ Name = 'bonsai-codec-probe'; Manifest = 'tools/codec-probe/Cargo.toml'; Group = 'Runtime'; Features = @('*') }
        @{ Name = 'prism-inference-server'; Manifest = 'server/Cargo.toml'; Group = 'Runtime'; Features = @('*') }
    )
}
