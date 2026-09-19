# Contributing

```sh
cargo test
cargo test --all-features
cargo run --example regularization
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
bash ci/check-package.sh
```

The package check verifies the release archive, then builds and runs a fresh
consumer with default features and with `metadata`. It resolves dependencies
without the repository lockfile. For local uncommitted changes, run
`bash ci/check-package.sh --allow-dirty`.

CI runs default and metadata tests on Linux, macOS, and Windows, plus a check
against the minimum supported Rhai version. Dependency ranges
stay within compatible release families: Rhai 1.x (from 1.16.2), SmartCore 0.3.x
(from 0.3.2), and Bincode 1.x (from 1.3.3). Upgrading SmartCore or the model encoding
requires explicit compatibility work.
