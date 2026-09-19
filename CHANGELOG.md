# Changelog

## 0.1.3 (2026-09-19)

- Bound dependency ranges to compatible release families. Fresh builds no longer
  select Bincode 3 or incompatible SmartCore versions.
- Validate training and prediction matrices, target lengths and types, finite
  values, and prediction feature counts before calling SmartCore. Include input
  locations in conversion errors.
- Accept integer and mixed numeric feature values and regression targets; retain
  integer labels for logistic regression.
- Reject underdetermined linear training before SmartCore's SVD solver can panic,
  and explain the sample-count requirement for both regression algorithms.
- Return serialization, deserialization, and backend errors through Rhai instead
  of unwrapping them. Reject non-finite regression predictions.
- Add behavioral coverage for all three algorithms, malformed inputs, model reuse,
  and Rhai error handling; make generated examples check actual predictions.
- Test default and metadata configurations across Linux, macOS, and Windows in CI,
  and verify a fresh consumer against the packaged crate.

The public `train` and `predict` signatures and default algorithm settings are
unchanged. Configurable training, evaluation utilities, and in-memory model
storage remain future work.
