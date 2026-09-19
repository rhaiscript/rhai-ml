use rhai::plugin::*;
use rhai::{Array, Dynamic, EvalAltResult, Map, Position, FLOAT};
use smartcorelib::linalg::basic::matrix::DenseMatrix;

fn error(message: impl std::fmt::Display) -> Box<EvalAltResult> {
    EvalAltResult::ErrorArithmetic(message.to_string(), Position::NONE).into()
}

fn number(value: &Dynamic, location: &str) -> Result<FLOAT, Box<EvalAltResult>> {
    let number = value
        .as_float()
        .or_else(|_| value.as_int().map(|value| value as FLOAT))
        .map_err(|_| error(format!("{location} must be a number (integer or float)")))?;
    if !number.is_finite() {
        return Err(error(format!("{location} must be finite")));
    }
    Ok(number)
}

fn training_alpha(algorithm: &str, options: &Map) -> Result<Option<FLOAT>, Box<EvalAltResult>> {
    for key in options.keys() {
        if algorithm == "linear" {
            return Err(error(format!("linear does not support option '{key}'")));
        }
        if key != "alpha" {
            return Err(error(format!(
                "unknown option '{key}' for {algorithm}; supported option: alpha"
            )));
        }
    }
    options
        .get("alpha")
        .map(|value| {
            let alpha = number(value, "options.alpha")?;
            if alpha < 0.0 {
                return Err(error("options.alpha must be nonnegative"));
            }
            Ok(alpha)
        })
        .transpose()
}

fn matrix(x: &Array) -> Result<(DenseMatrix<FLOAT>, usize), Box<EvalAltResult>> {
    if x.is_empty() {
        return Err(error("x must contain at least one row"));
    }
    let mut rows = Vec::with_capacity(x.len());
    let mut columns = 0;
    for (row_index, observation) in x.iter().enumerate() {
        let row = observation
            .clone()
            .into_array()
            .map_err(|_| error(format!("x[{row_index}] must be an array")))?;
        if row_index == 0 {
            columns = row.len();
            if columns == 0 {
                return Err(error("x rows must contain at least one feature"));
            }
        }
        if row.len() != columns {
            return Err(error(format!(
                "x[{row_index}] has {} features; expected {columns}",
                row.len()
            )));
        }
        let values = row
            .iter()
            .enumerate()
            .map(|(column, value)| number(value, &format!("x[{row_index}][{column}]")))
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(values);
    }
    // SmartCore 0.3 requires a nonempty, rectangular matrix; validated above.
    Ok((DenseMatrix::from_2d_vec(&rows), columns))
}

fn regression_targets(y: &Array) -> Result<Vec<FLOAT>, Box<EvalAltResult>> {
    y.iter()
        .enumerate()
        .map(|(index, value)| number(value, &format!("y[{index}]")))
        .collect()
}

fn predictions(values: Vec<FLOAT>) -> Result<Array, Box<EvalAltResult>> {
    values
        .into_iter()
        .map(|value| {
            if value.is_finite() {
                Ok(Dynamic::from_float(value))
            } else {
                Err(error("prediction produced a non-finite value"))
            }
        })
        .collect()
}

/// Training and prediction functions exposed to Rhai.
#[export_module]
pub mod train_and_predict_functions {
    use super::{error, matrix, predictions, regression_targets, training_alpha};
    use rhai::{Array, Dynamic, EvalAltResult, ImmutableString, Map, FLOAT, INT};
    use smartcorelib::{
        linalg::basic::matrix::DenseMatrix,
        linear::{
            lasso::{Lasso, LassoParameters},
            linear_regression::{LinearRegression, LinearRegressionParameters},
            logistic_regression::{LogisticRegression, LogisticRegressionParameters},
        },
    };

    /// An opaque trained model, including its expected number of input features.
    #[derive(Clone, Default)]
    pub struct Model {
        saved_model: Vec<u8>,
        model_type: String,
        features: usize,
    }

    /// Trains a [`smartcore`](https://smartcorelib.org/) machine learning model.
    /// Use [`predict`](#predictx-array-model-model---array) to make predictions.
    /// Available model types are:
    /// 1. `linear` - ordinary least squares linear regression
    /// 2. `logistic` - logistic regression with integer class labels
    /// 3. `lasso` - lasso regression
    ///
    /// `x` must be a nonempty rectangular array of finite numbers. Regression
    /// targets may be integers or floats; logistic targets must be integers.
    /// The number of targets must equal the number of rows in `x`. Linear and
    /// lasso regression require more rows than features. Input validation and
    /// backend errors are reported as Rhai errors.
    /// ```typescript
    /// let xdata = [[0.0], [1.0], [2.0], [3.0]];
    /// let ydata = [1.0, 3.0, 5.0, 7.0];
    /// let model = train(xdata, ydata, "linear");
    /// let ypred = predict([[4.0]], model);
    /// abs(ypred[0] - 9.0) < 0.000001;
    /// ```
    #[rhai_fn(name = "train", return_raw, pure)]
    pub fn train_model(
        x: &mut Array,
        y: Array,
        algorithm: ImmutableString,
    ) -> Result<Model, Box<EvalAltResult>> {
        train_model_with_options(x, y, algorithm, Map::new())
    }

    /// Trains a model with an options map. Lasso and logistic regression accept
    /// `alpha`, a finite nonnegative integer or float controlling regularization.
    /// Higher values apply a stronger penalty. The defaults are `1.0` for lasso
    /// and `0.0` for logistic regression. Omitting `alpha`, or passing an empty
    /// map, preserves the defaults of the three-argument `train` call.
    ///
    /// Linear regression accepts only an empty map. Unknown options and invalid
    /// values return Rhai errors. The input requirements are the same as for
    /// the three-argument call.
    /// ```typescript
    /// let x = [[0], [1], [2], [3], [4], [5]];
    /// let y = [1, 3, 5, 7, 9, 11];
    /// let model = train(x, y, "lasso", #{ alpha: 0.1 });
    /// abs(predict([[6]], model)[0] - 13.0) < 0.2;
    /// ```
    #[rhai_fn(name = "train", return_raw, pure)]
    pub fn train_model_with_options(
        x: &mut Array,
        y: Array,
        algorithm: ImmutableString,
        options: Map,
    ) -> Result<Model, Box<EvalAltResult>> {
        let algorithm = algorithm.as_str();
        if !matches!(algorithm, "linear" | "lasso" | "logistic") {
            return Err(error(format!(
                "{algorithm} is not a recognized model type; expected linear, lasso, or logistic"
            )));
        }
        let alpha = training_alpha(algorithm, &options)?;
        let (xvec, features) = matrix(x)?;
        if y.len() != x.len() {
            return Err(error(format!(
                "y has {} targets; expected {} (one per row of x)",
                y.len(),
                x.len()
            )));
        }
        let saved_model = match algorithm {
            "linear" => {
                let yvec = regression_targets(&y)?;
                // The SVD solver needs at least as many rows as columns, including
                // the intercept column that SmartCore adds to the design matrix.
                if x.len() <= features {
                    return Err(error("linear training requires more rows than features"));
                }
                let model =
                    LinearRegression::fit(&xvec, &yvec, LinearRegressionParameters::default())
                        .map_err(error)?;
                bincode::serialize(&model).map_err(error)?
            }
            "lasso" => {
                let yvec = regression_targets(&y)?;
                if x.len() <= features {
                    return Err(error("lasso training requires more rows than features"));
                }
                let mut parameters = LassoParameters::default();
                if let Some(alpha) = alpha {
                    // SmartCore's lasso parameter is f64 even when Rhai uses f32.
                    #[allow(clippy::unnecessary_cast)]
                    {
                        parameters.alpha = alpha as f64;
                    }
                }
                let model = Lasso::fit(&xvec, &yvec, parameters).map_err(error)?;
                bincode::serialize(&model).map_err(error)?
            }
            "logistic" => {
                let yvec = y
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        value.as_int().map_err(|_| {
                            error(format!(
                                "y[{index}] must be an integer class label for logistic"
                            ))
                        })
                    })
                    .collect::<Result<Vec<INT>, _>>()?;
                let mut parameters = LogisticRegressionParameters::default();
                if let Some(alpha) = alpha {
                    parameters.alpha = alpha;
                }
                let model = LogisticRegression::fit(&xvec, &yvec, parameters).map_err(error)?;
                bincode::serialize(&model).map_err(error)?
            }
            _ => return Err(error("unrecognized model type")),
        };
        Ok(Model {
            saved_model,
            model_type: algorithm.to_owned(),
            features,
        })
    }

    /// Predicts dependent variables with a model produced by `train`.
    /// `x` must be a nonempty rectangular array of finite numbers with the same
    /// number of features as the training data. Regression returns floats;
    /// logistic regression returns integer class labels.
    /// ```typescript
    /// let model = train([[0], [1], [2], [3]], [1, 3, 5, 7], "linear");
    /// let ypred = predict([[4], [5]], model);
    /// abs(ypred[0] - 9.0) < 0.000001 && abs(ypred[1] - 11.0) < 0.000001;
    /// ```
    #[rhai_fn(name = "predict", return_raw, pure)]
    pub fn predict_with_model(x: &mut Array, model: Model) -> Result<Array, Box<EvalAltResult>> {
        let (xvec, features) = matrix(x)?;
        if features != model.features {
            return Err(error(format!(
                "x has {features} features; model expects {}",
                model.features
            )));
        }
        match model.model_type.as_str() {
            "linear" => {
                let model_ready: LinearRegression<FLOAT, FLOAT, DenseMatrix<FLOAT>, Vec<FLOAT>> =
                    bincode::deserialize(&model.saved_model).map_err(error)?;
                predictions(model_ready.predict(&xvec).map_err(error)?)
            }
            "lasso" => {
                let model_ready: Lasso<FLOAT, FLOAT, DenseMatrix<FLOAT>, Vec<FLOAT>> =
                    bincode::deserialize(&model.saved_model).map_err(error)?;
                predictions(model_ready.predict(&xvec).map_err(error)?)
            }
            "logistic" => {
                let model_ready: LogisticRegression<FLOAT, INT, DenseMatrix<FLOAT>, Vec<INT>> =
                    bincode::deserialize(&model.saved_model).map_err(error)?;
                Ok(model_ready
                    .predict(&xvec)
                    .map_err(error)?
                    .into_iter()
                    .map(Dynamic::from_int)
                    .collect())
            }
            algorithm => Err(error(format!("{algorithm} is not a recognized model type"))),
        }
    }
}
