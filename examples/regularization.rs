fn main() {
    let compared = rhai_ml::eval::<bool>(
        r#"
        fn mean_squared_error(actual, expected) {
            let total = 0.0;
            for i in 0..actual.len() {
                let residual = actual[i] - expected[i];
                total += residual * residual;
            }
            total / actual.len()
        }

        // Fixed training and held-out validation samples from y = 2*x + 1.
        let x_train = [[0], [1], [2], [3], [4], [5]];
        let y_train = [1, 3, 5, 7, 9, 11];
        let x_validation = [[6], [7]];
        let y_validation = [13, 15];
        let errors = [];

        for alpha in [0.1, 1.0] {
            let model = train(x_train, y_train, "lasso", #{ alpha: alpha });
            let predictions = predict(x_validation, model);
            let mse = mean_squared_error(predictions, y_validation);
            print(`alpha=${alpha}: validation MSE=${mse}`);
            errors.push(mse);
        }

        // Less shrinkage helps on this noiseless toy dataset. Other data can
        // favor a stronger penalty; choose using validation data, not training fit.
        errors[0] < errors[1]
        "#,
    )
    .expect("regularization comparison should succeed");
    assert!(compared);
}
