use rhai::{packages::Package, Array, Dynamic, Engine, Scope, FLOAT, INT};
use rhai_ml::{eval, MLPackage};

fn engine() -> Engine {
    let mut engine = Engine::new();
    engine.register_global_module(MLPackage::new().as_shared_module());
    engine
}

fn assert_script_error(script: &str, expected: &str) {
    // A Rust panic fails the test; the contract is an ordinary Rhai error.
    let error = engine()
        .eval::<Dynamic>(script)
        .expect_err("script should return a Rhai error");
    assert!(
        error.to_string().contains(expected),
        "expected {expected:?}, got {error} for {script}"
    );
}

#[test]
fn linear_predicts_held_out_values_with_mixed_numeric_inputs() {
    let values = engine()
        .eval::<Array>(
            r#"
        let model = train([[0], [1.0], [2], [3.0]], [1, 3.0, 5, 7.0], "linear");
        predict([[4], [5.0]], model)
        "#,
        )
        .unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([9.0, 11.0]) {
        assert!((actual.as_float().unwrap() - expected).abs() < 1e-8);
    }
}

#[test]
fn lasso_predicts_held_out_values() {
    let values = engine()
        .eval::<Array>(
            r#"
        let model = train([[0], [1], [2], [3], [4], [5]], [1, 3, 5, 7, 9, 11], "lasso");
        predict([[6], [7]], model)
        "#,
        )
        .unwrap();
    assert_eq!(values.len(), 2);
    // Default L1 regularization shrinks the slope, so use a bounded prediction error.
    for (actual, expected) in values.iter().zip([13.0, 15.0]) {
        assert!((actual.as_float().unwrap() - expected).abs() < 2.0);
    }
}

#[test]
fn logistic_preserves_integer_class_labels() {
    let values = engine()
        .eval::<Array>(
            r#"
        let model = train([[-3], [-2], [-1], [1], [2], [3]], [-5, -5, -5, 7, 7, 7], "logistic");
        predict([[-4], [4]], model)
        "#,
        )
        .unwrap();
    let labels: Vec<INT> = values.iter().map(|value| value.as_int().unwrap()).collect();
    assert_eq!(labels, [-5, 7]);
}

#[test]
fn model_can_be_reused_without_mutating_inputs() {
    assert!(eval::<bool>(
        r#"
        let x = [[0], [1], [2], [3]];
        let y = [1, 3, 5, 7];
        let model = train(x, y, "linear");
        let query = [[4]];
        let first = predict(query, model);
        let second = predict(query, model);
        x == [[0], [1], [2], [3]] && y == [1, 3, 5, 7]
            && query == [[4]] && first == second
        "#
    )
    .unwrap());
}

#[test]
fn malformed_training_matrices_return_errors_for_every_algorithm() {
    for algorithm in ["linear", "lasso", "logistic"] {
        for (matrix, expected) in [
            ("[]", "at least one row"),
            ("[[]]", "at least one feature"),
            ("[1, 2]", "x[0] must be an array"),
            ("[[1], 2]", "x[1] must be an array"),
            ("[[1, 2], [3]]", "x[1] has 1 features; expected 2"),
            ("[[1], [2, 3]]", "x[1] has 2 features; expected 1"),
            ("[[1], []]", "x[1] has 0 features; expected 1"),
            ("[[true], [2]]", "x[0][0] must be a number"),
            (r#"[[1], ["bad"]]"#, "x[1][0] must be a number"),
            ("[[[1]], [2]]", "x[0][0] must be a number"),
        ] {
            assert_script_error(
                &format!(r#"train({matrix}, [0, 1], "{algorithm}")"#),
                expected,
            );
        }
    }
}

#[test]
fn target_lengths_must_match_training_rows() {
    for algorithm in ["linear", "lasso", "logistic"] {
        for targets in ["[]", "[0]", "[0, 1, 2]"] {
            assert_script_error(
                &format!(r#"train([[0], [1]], {targets}, "{algorithm}")"#),
                "expected 2 (one per row of x)",
            );
        }
    }
}

#[test]
fn regression_targets_must_be_numeric() {
    for algorithm in ["linear", "lasso"] {
        for targets in [r#"[1, "bad", 3]"#, "[1, true, 3]", "[1, [], 3]"] {
            assert_script_error(
                &format!(r#"train([[0], [1], [2]], {targets}, "{algorithm}")"#),
                "y[1] must be a number",
            );
        }
    }
}

#[test]
fn logistic_requires_integer_class_labels() {
    for targets in ["[0.0, 1.0]", "[0, 1.0]", "[false, true]", r#"["a", "b"]"#] {
        assert_script_error(
            &format!(r#"train([[0], [1]], {targets}, "logistic")"#),
            "must be an integer class label",
        );
    }
}

#[test]
fn unknown_algorithms_return_a_useful_error() {
    assert_script_error(
        r#"train([[0], [1]], [0, 1], "unknown")"#,
        "expected linear, lasso, or logistic",
    );
}

#[test]
fn malformed_prediction_matrices_return_errors() {
    for algorithm in ["linear", "lasso", "logistic"] {
        for (query, expected) in [
            ("[]", "at least one row"),
            ("[[]]", "at least one feature"),
            ("[1]", "x[0] must be an array"),
            ("[[1], [2, 3]]", "x[1] has 2 features; expected 1"),
            (r#"[["bad"]]"#, "x[0][0] must be a number"),
            ("[[1, 2]]", "x has 2 features; model expects 1"),
        ] {
            assert_script_error(
                &format!(
                    r#"
                let model = train([[0], [1], [2], [3]], [0, 0, 1, 1], "{algorithm}");
                predict({query}, model)
                "#
                ),
                expected,
            );
        }
    }
}

#[test]
fn predictions_reject_too_few_features() {
    assert_script_error(
        r#"
        let model = train([[0, 0], [1, 0], [0, 1], [1, 1]], [1, 3, 4, 6], "linear");
        predict([[1]], model)
        "#,
        "x has 1 features; model expects 2",
    );
}

#[test]
fn non_finite_numbers_return_errors() {
    let mut engine = Engine::new();
    engine.register_global_module(MLPackage::new().as_shared_module());
    for value in [FLOAT::NAN, FLOAT::INFINITY, FLOAT::NEG_INFINITY] {
        for (script, expected) in [
            (
                r#"train([[bad], [1]], [0, 1], "linear")"#,
                "x[0][0] must be finite",
            ),
            (
                r#"train([[0], [1]], [0, bad], "linear")"#,
                "y[1] must be finite",
            ),
            (
                r#"train([[0], [1]], [0, bad], "lasso")"#,
                "y[1] must be finite",
            ),
            (
                r#"train([[bad], [1]], [0, 1], "logistic")"#,
                "x[0][0] must be finite",
            ),
            (
                r#"let model = train([[0], [1], [2]], [0, 1, 2], "linear"); predict([[bad]], model)"#,
                "x[0][0] must be finite",
            ),
        ] {
            let mut scope = Scope::new();
            scope.push("bad", value);
            let error = engine
                .eval_with_scope::<Dynamic>(&mut scope, script)
                .unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }
}

#[test]
fn backend_fit_errors_reach_the_script() {
    assert_script_error(
        r#"train([[0], [1]], [7, 7], "logistic")"#,
        "number of classes",
    );
    assert_script_error(
        r#"train([[1], [1], [1]], [1, 2, 3], "lasso")"#,
        "constant column",
    );
    assert_script_error(
        r#"train([[1, 2]], [1], "lasso")"#,
        "more rows than features",
    );
}

#[test]
fn validation_errors_can_be_caught_in_rhai() {
    assert!(eval::<bool>(
        r#"
        let caught = false;
        try { train([[true]], [1], "linear"); }
        catch (err) { caught = true; }
        caught
        "#,
    )
    .unwrap());
}

#[test]
fn underdetermined_regression_training_returns_an_error() {
    for algorithm in ["linear", "lasso"] {
        for (x, y) in [
            ("[[1, 2]]", "[1]"),
            ("[[1]]", "[1]"),
            ("[[1, 2], [3, 4]]", "[1, 2]"),
        ] {
            assert_script_error(
                &format!(r#"train({x}, {y}, "{algorithm}")"#),
                "more rows than features",
            );
        }
    }
}

#[test]
fn rank_deficient_linear_data_still_predicts() {
    let value = eval::<FLOAT>(
        r#"
        let model = train([[1, 2], [2, 3], [3, 4]], [1, 2, 3], "linear");
        predict([[4, 5]], model)[0]
        "#,
    )
    .unwrap();
    assert!((value - 4.0).abs() < 1e-8);
}

#[test]
fn prediction_overflow_is_reported_as_an_error() {
    assert_script_error(
        r#"
        let model = train([[0], [1], [2]], [0.0, 1.0e150, 2.0e150], "linear");
        predict([[1.0e200]], model)
        "#,
        "prediction produced a non-finite value",
    );
}
