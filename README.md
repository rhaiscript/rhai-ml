[![tests](https://github.com/rhaiscript/rhai-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/rhaiscript/rhai-ml/actions/workflows/tests.yml)
[![Crates.io](https://img.shields.io/crates/v/rhai-ml.svg)](https://crates.io/crates/rhai-ml)
[![docs.rs](https://img.shields.io/docsrs/rhai-ml/latest?logo=rust)](https://docs.rs/rhai-ml)

# rhai-ml

Machine learning for the [Rhai](https://rhai.rs/) scripting language, backed by
[SmartCore](https://smartcorelib.org/). Train linear, lasso, and logistic regression
models directly in your scripts.

## Install

```toml
rhai-ml = "0.1.3"
```

## Usage

```rust
use rhai_ml::eval;

let prediction = eval::<f64>(r#"
    let model = train([[0], [1], [2], [3]], [1, 3, 5, 7], "linear");
    predict([[4]], model)[0]
"#).unwrap();

assert!((prediction - 9.0).abs() < 0.000001);
```

To add the package to an existing Rhai engine:

```rust
use rhai::{packages::Package, Engine};
use rhai_ml::MLPackage;

let mut engine = Engine::new();
engine.register_global_module(MLPackage::new().as_shared_module());
```

Inputs must be nonempty rectangular arrays of finite numbers, with one target per
training row. Logistic regression uses integer class labels. Invalid inputs return
Rhai errors that scripts can handle with `try`/`catch`.

The optional `metadata` feature generates API documentation and tests the Rhai
examples. Run `cargo run --example quickstart` to try all three algorithms.

See the [API reference](https://docs.rs/rhai-ml),
[input rules and errors](https://github.com/rhaiscript/rhai-ml/blob/master/docs/usage.md),
[development checks](https://github.com/rhaiscript/rhai-ml/blob/master/CONTRIBUTING.md),
and [changelog](https://github.com/rhaiscript/rhai-ml/blob/master/CHANGELOG.md).
