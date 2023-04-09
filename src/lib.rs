#![warn(clippy::all)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
#![doc = include_str!(concat!(env!("OUT_DIR"), "/rhai-ml-docs.md"))]
#![doc = include_str!("../docs/highlight.html")]

use rhai::{def_package, packages::Package, plugin::*, Engine, EvalAltResult};
mod train_and_predict;
use train_and_predict::train_and_predict_functions;

def_package! {
    /// Package for machine learning
    pub MLPackage(lib) {

        combine_with_exported_module!(lib, "rhai_ml_train_and_predict", train_and_predict_functions);
    }
}

/// This provides the ability to easily evaluate a line (or lines) of code without explicitly
/// setting up a script engine
/// ```
/// use rhai_ml::eval;
/// use rhai::FLOAT;
/// print!("{:?}", eval::<FLOAT>("let x = 5;"));
/// ```
pub fn eval<T: Clone + std::marker::Send + std::marker::Sync + 'static>(
    script: &str,
) -> Result<T, Box<EvalAltResult>> {
    let mut engine = Engine::new();
    engine.register_global_module(MLPackage::new().as_shared_module());
    engine.eval::<T>(script)
}
