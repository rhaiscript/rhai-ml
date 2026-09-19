fn main() {
    let passed = rhai_ml::eval::<bool>(
        r#"
        let linear = train([[0], [1], [2], [3]], [1, 3, 5, 7], "linear");
        let linear_ok = abs(predict([[4]], linear)[0] - 9.0) < 0.000001;

        let lasso = train([[0], [1], [2], [3], [4], [5]], [1, 3, 5, 7, 9, 11], "lasso");
        let lasso_ok = abs(predict([[6]], lasso)[0] - 13.0) < 2.0;

        let logistic = train([[-3], [-2], [-1], [1], [2], [3]], [0, 0, 0, 1, 1, 1], "logistic");
        let logistic_ok = predict([[-4], [4]], logistic) == [0, 1];

        let caught = false;
        try { predict([[1, 2]], linear); }
        catch (err) { caught = true; }
        linear_ok && lasso_ok && logistic_ok && caught
        "#,
    )
    .expect("quickstart script should succeed");
    assert!(passed);
    println!("Training, prediction, and error handling passed for the packaged crate.");
}
