mod sgd;
use std::collections::HashMap;

use af::{self, Array, NormType};
use arrayfire::Dim4;

use crate::error::HALError;
use crate::params::ParamManager;

pub use self::sgd::SGD;

pub trait Optimizer {
    fn new(params: &HashMap<&str, &str>) -> Self
    where
        Self: Sized;

    fn setup(&mut self, dims: Vec<Dim4>);

    fn update(&mut self, parameter_manager: &mut ParamManager, batch_size: u64);
}

pub fn get_optimizer_with_defaults(name: &str) -> Result<Box<dyn Optimizer>, HALError> {
    match name.to_lowercase().as_str() {
        "sgd" => Ok(Box::new(SGD::default())),
        _ => Err(HALError::UNKNOWN),
    }
}

// pub fn clip_grad(input: &Array<f32>, rescale: f32) -> Array<f32> {
//     let norm = af::norm(input, NormType::VECTOR_2, 0f64, 0f64) as f32;
//     let scale = rescale / norm.max(rescale);

// }
