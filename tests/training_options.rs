use rhai::{packages::Package, Array, Dynamic, Engine, Scope, FLOAT, INT};
use rhai_ml::{eval, MLPackage};

fn engine() -> Engine {
    let mut engine = Engine::new();
    engine.register_global_module(MLPackage::new().as_shared_module());
    engine
}

fn assert_error(script: &str, expected: &str) {
    let error = engine().eval::<Dynamic>(script).unwrap_err();
    assert!(error.to_string().contains(expected), "{script}: {error}");
}

#[test]
fn empty_options_preserve_every_algorithms_predictions() {
    for algorithm in ["linear", "lasso", "logistic"] {
        assert!(eval::<bool>(&format!(
            r#"
            let x = [[0], [1], [2], [3], [4], [5]];
            let y = [0, 0, 0, 0, 1, 1];
            let original = train(x, y, "{algorithm}");
            let explicit = train(x, y, "{algorithm}", #{{}});
            predict([[0], [3], [6]], original) == predict([[0], [3], [6]], explicit)
            "#
        ))
        .unwrap());
    }
}

#[test]
fn explicit_default_alphas_match_existing_calls() {
    for (algorithm, alpha) in [("lasso", 1.0), ("logistic", 0.0)] {
        assert!(eval::<bool>(&format!(
            r#"
            let x = [[0], [1], [2], [3], [4], [5]];
            let y = [0, 0, 0, 0, 1, 1];
            let original = train(x, y, "{algorithm}");
            let explicit = train(x, y, "{algorithm}", #{{ alpha: {alpha:.1} }});
            predict([[0], [3], [6]], original) == predict([[0], [3], [6]], explicit)
            "#
        ))
        .unwrap());
    }
}

#[test]
fn lasso_alpha_changes_held_out_predictions() {
    let predictions = engine()
        .eval::<Array>(
            r#"
            let x = [[0], [1], [2], [3], [4], [5]];
            let y = [1, 3, 5, 7, 9, 11];
            let weak = train(x, y, "lasso", #{ alpha: 0.1 });
            let strong = train(x, y, "lasso", #{ alpha: 1.0 });
            [predict([[6]], weak)[0], predict([[6]], strong)[0]]
            "#,
        )
        .unwrap();
    let weak = predictions[0].as_float().unwrap();
    let strong = predictions[1].as_float().unwrap();
    assert!((weak - 13.0).abs() < 0.2, "weak prediction: {weak}");
    assert!(weak - strong > 0.5, "weak: {weak}, strong: {strong}");
}

#[test]
fn logistic_alpha_changes_predictions_and_preserves_class_labels() {
    // An imbalanced sample makes strong regularization favor the majority class.
    let predictions = engine()
        .eval::<Array>(
            r#"
            let x = [[0], [1], [2], [3], [4], [5]];
            let y = [-5, -5, -5, -5, 7, 7];
            let weak = train(x, y, "logistic", #{ alpha: 0.0 });
            let strong = train(x, y, "logistic", #{ alpha: 100.0 });
            [predict([[6]], weak)[0], predict([[6]], strong)[0]]
            "#,
        )
        .unwrap();
    let labels: Vec<INT> = predictions
        .iter()
        .map(|value| value.as_int().unwrap())
        .collect();
    assert_eq!(labels, [7, -5]);
}

#[test]
fn integer_and_float_alphas_are_equivalent() {
    for algorithm in ["lasso", "logistic"] {
        assert!(eval::<bool>(&format!(
            r#"
            let x = [[0], [1], [2], [3], [4], [5]];
            let y = [0, 0, 0, 0, 1, 1];
            let integer = train(x, y, "{algorithm}", #{{ alpha: 1 }});
            let floating = train(x, y, "{algorithm}", #{{ alpha: 1.0 }});
            predict([[0], [6]], integer) == predict([[0], [6]], floating)
            "#
        ))
        .unwrap());
    }
}

#[test]
fn zero_alpha_is_supported() {
    let prediction = eval::<FLOAT>(
        r#"
        let model = train([[0], [1], [2], [3]], [1, 3, 5, 7], "lasso", #{ alpha: 0 });
        predict([[4]], model)[0]
        "#,
    )
    .unwrap();
    assert!((prediction - 9.0).abs() < 1e-4, "{prediction}");
}

#[test]
fn invalid_alpha_types_and_negative_values_return_errors() {
    for algorithm in ["lasso", "logistic"] {
        for (value, message) in [
            ("-1", "options.alpha must be nonnegative"),
            ("-0.1", "options.alpha must be nonnegative"),
            (r#""0.1""#, "options.alpha must be a number"),
            ("true", "options.alpha must be a number"),
            ("[]", "options.alpha must be a number"),
            ("#{}", "options.alpha must be a number"),
            ("()", "options.alpha must be a number"),
        ] {
            assert_error(
                &format!(
                    r#"train([[0], [1], [2]], [0, 0, 1], "{algorithm}", #{{ alpha: {value} }})"#
                ),
                message,
            );
        }
    }
}

#[test]
fn non_finite_alpha_values_return_errors() {
    let engine = engine();
    for algorithm in ["lasso", "logistic"] {
        for alpha in [FLOAT::NAN, FLOAT::INFINITY, FLOAT::NEG_INFINITY] {
            let mut scope = Scope::new();
            scope.push("alpha", alpha);
            let error = engine
                .eval_with_scope::<Dynamic>(
                    &mut scope,
                    &format!(
                        r#"train([[0], [1], [2]], [0, 0, 1], "{algorithm}", #{{ alpha: alpha }})"#
                    ),
                )
                .unwrap_err();
            assert!(
                error.to_string().contains("options.alpha must be finite"),
                "{error}"
            );
        }
    }
}

#[test]
fn unknown_options_are_never_ignored() {
    for algorithm in ["lasso", "logistic"] {
        for options in ["#{ alhpa: 0.1 }", "#{ alpha: 0.1, alhpa: 0.1 }"] {
            assert_error(
                &format!(r#"train([[0], [1], [2]], [0, 0, 1], "{algorithm}", {options})"#),
                "unknown option 'alhpa'",
            );
        }
    }
    for options in ["#{ alpha: 0.1 }", "#{ solver: \"SVD\" }"] {
        assert_error(
            &format!(r#"train([[0], [1], [2]], [0, 0, 1], "linear", {options})"#),
            "linear does not support option",
        );
    }
}

#[test]
fn overload_retains_input_and_algorithm_validation() {
    for algorithm in ["linear", "lasso", "logistic"] {
        for (x, y, message) in [
            ("[]", "[]", "at least one row"),
            ("[[0], [1, 2]]", "[0, 1]", "features; expected"),
            ("[[0], [1]]", "[0]", "one per row of x"),
        ] {
            assert_error(
                &format!(r#"train({x}, {y}, "{algorithm}", #{{}})"#),
                message,
            );
        }
    }
    assert_error(
        r#"train([[0], [1]], [0, 1], "unknown", #{ alpha: 1 })"#,
        "not a recognized model type",
    );
}

#[test]
fn options_and_models_can_be_reused_without_mutation() {
    assert!(eval::<bool>(
        r#"
        let x = [[0], [1], [2], [3], [4], [5]];
        let y = [1, 3, 5, 7, 9, 11];
        let options = #{ alpha: 0.1 };
        let first = train(x, y, "lasso", options);
        let second = train(x, y, "lasso", options);
        let predictions = predict([[6]], first);
        predictions == predict([[6]], first) && predictions == predict([[6]], second)
            && options == #{ alpha: 0.1 } && x == [[0], [1], [2], [3], [4], [5]]
            && y == [1, 3, 5, 7, 9, 11]
        "#,
    )
    .unwrap());
}

#[test]
fn option_errors_can_be_caught_by_scripts() {
    assert!(eval::<bool>(
        r#"
        let caught = false;
        try { train([[0], [1], [2]], [0, 0, 1], "lasso", #{ alhpa: 0.1 }); }
        catch (err) { caught = true; }
        caught
        "#,
    )
    .unwrap());
}
