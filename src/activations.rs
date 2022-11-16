use af::{self, Dim4};
use arrayfire::Array;

use crate::{error::HALError, utils};

/// Returns the tanh activated value
pub fn tanh(x: &Array<f32>) -> Array<f32> {
    af::tanh(x)
}

/// Returns the derivative of tanh [assumes that tanh has already been applied]
/// 1 - tanh(x) * tanh(x)
pub fn tanh_derivative(x: &Array<f32>) -> Array<f32> {
    let one = utils::constant(x.dims(), 1.0f32);
    let grad = af::sub(&one, &af::mul(x, x, false), false);
    grad
}

pub fn get_activation(name: &str, x: &Array<f32>) -> Result<Array<f32>, HALError> {
    match name {
        "tanh" => Ok(tanh(x)),
        _ => Err(HALError::UNKNOWN),
    }
}

pub fn get_derivative(name: &str, x: &Array<f32>) -> Result<Array<f32>, HALError> {
    match name {
        "tanh" => Ok(tanh_derivative(x)),
        _ => Err(HALError::UNKNOWN),
    }
}
