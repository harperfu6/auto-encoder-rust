mod sgd;
use std::collections::HashMap;

use af::Array;
use af::{self, NormType};
use arrayfire::Dim4;

use crate::params::ParamManager;

pub use self::sgd::SGD;

pub trait Optimizer {
    fn new(params: &HashMap<&str, &str>) -> Self
    where
        Self: Sized;

    fn setup(&mut self, dims: Vec<Dim4>);

    fn update(&mut self, parameter_manager: &mut ParamManager, batch_size: u64);
}

// pub fn clip_grad(input: &Array<f32>, rescale: f32) -> Array<f32> {
//     let norm = af::norm(input, NormType::VECTOR_2, 0f64, 0f64) as f32;
//     let scale = rescale / norm.max(rescale);

// }
