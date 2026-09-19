# Inputs and errors

`train(x, y, algorithm)` accepts `linear`, `lasso`, and `logistic`.

- `x` must contain at least one row and one feature. Every row must have the same
  length, and every value must be a finite integer or float. Integers are converted
  to Rhai's floating-point type for computation.
- `y` must have one target per row. Linear and lasso targets may be integers or
  floats; logistic targets must be integer class labels with at least two classes.
- Linear and lasso training require more rows than features. Lasso uses
  normalization by default, so constant feature columns are rejected by SmartCore.
- `predict(x, model)` uses the same matrix rules and requires the feature count
  used during training. Regression returns floats; classification returns integer
  class labels. Non-finite regression predictions are reported as errors.

Validation failures and errors returned by SmartCore become Rhai errors. Scripts
can handle them with `try`/`catch`; Rust callers receive `Err` from evaluation.
For example:

```rhai
let model = train([[0], [1], [2], [3]], [1, 3, 5, 7], "linear");
try {
    predict([[1, 2]], model);
} catch (err) {
    print(err); // x has 2 features; model expects 1
}
```

Run `cargo run --example quickstart` for an example covering all three algorithms,
held-out predictions, and script error handling.


## Rhai metadata

The optional `metadata` feature generates the script API reference and runs the
Rhai examples as tests. Behavioral tests run with or without this feature.
