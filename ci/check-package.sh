#!/usr/bin/env bash
# Run with --allow-dirty to verify an uncommitted release candidate locally.
set -euo pipefail

cd "$(dirname "$0")/.."
cargo package --all-features "$@"
read -r crate_name crate_version target_dir < <(
  cargo metadata --format-version 1 --no-deps |
    python3 -c 'import json,sys; data=json.load(sys.stdin); package=data["packages"][0]; print(package["name"], package["version"], data["target_directory"])'
)
probe_dir=$(mktemp -d)
trap 'rm -rf "$probe_dir"' EXIT

tar -xzf "$target_dir/package/$crate_name-$crate_version.crate" -C "$probe_dir"
mkdir -p "$probe_dir/consumer/src"
cat > "$probe_dir/consumer/Cargo.toml" <<TOML
[package]
name = "rhai-ml-package-smoke"
version = "0.0.0"
edition = "2021"

[features]
metadata = ["rhai-ml/metadata"]

[dependencies]
rhai-ml = { path = "../$crate_name-$crate_version" }
TOML
cp examples/quickstart.rs "$probe_dir/consumer/src/main.rs"
# This consumer starts without a lockfile and depends only on the packaged source.
export CARGO_TARGET_DIR="$target_dir/package-smoke"
cargo run --manifest-path "$probe_dir/consumer/Cargo.toml"
cargo run --manifest-path "$probe_dir/consumer/Cargo.toml" --features metadata
