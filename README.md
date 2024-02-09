[![tests](https://github.com/rhaiscript/rhai-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/rhaiscript/rhai-ml/actions/workflows/tests.yml)
[![Crates.io](https://img.shields.io/crates/v/rhai-ml.svg)](https://crates.io/crates/rhai-ml)
[![docs.rs](https://img.shields.io/docsrs/rhai-ml/latest?logo=rust)](https://docs.rs/rhai-ml)

# About `rhai-ml`

This crate provides some basic machine learning and artificial intelligence utilities for the [`Rhai`](https://rhai.rs/) 
scripting language. For a complete API reference, check [the docs](https://docs.rs/rhai-ml).

# Install

To use the latest released version of `rhai-ml`, add this to your `Cargo.toml`:

```toml
rhai-ml = "0.1.2"
```

To use the bleeding edge instead, add this:

```toml
rhai-ml = { git = "https://github.com/cmccomb/rhai-ml" }
```

# Usage

Using this crate is pretty simple! If you just want to evaluate a single line of [`Rhai`](https://rhai.rs/), then you only need:

```rust
use rhai::FLOAT;
use rhai_ml::eval;
let result = eval::<FLOAT>("\
let xdata = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]; \
let ydata = [1.0, 2.0, 3.0]; \
let model = train(xdata, ydata, \"linear\"); \
let ypred = predict(xdata, model);
ypred[0]
").unwrap();
```

If you need to use `rhai-ml` as part of a persistent [`Rhai`](https://rhai.rs/) scripting engine, then do this instead:

```rust
use rhai::{Engine, packages::Package, FLOAT};
use rhai_ml::MLPackage;

// Create a new Rhai engine
let mut engine = Engine::new();

// Add the rhai-ml package to the new engine
engine.register_global_module(MLPackage::new().as_shared_module());

// Now run your code
let value = engine.eval::<FLOAT>("\
let xdata = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]; \
let ydata = [1.0, 2.0, 3.0]; \
let model = train(xdata, ydata, \"linear\"); \
let ypred = predict(xdata, model);
ypred[0]
").unwrap();
```

# Features

| Feature     | Default  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `metadata`  | Disabled | Enables exporting function metadata and is ___necessary for running doc-tests on Rhai examples___.                                                                                                                                                                                                                                                                                                                                                                                                                    |
