use af;
use af::Array;

use crate::error::HALError;

/// Return a vector form of the l2 error
/// (y - x) * (y - x)
pub fn l2_vec(pred: &Array<f32>, target: &Array<f32>) -> Array<f32> {
    let diff = af::sub(pred, target, false);
    af::mul(&diff, &diff, false)
}

/// Provide a reduced form the mean squared error loss (single scalar)
pub fn mse(pred: &Array<f32>, target: &Array<f32>) -> f32 {
    0.5f32 * af::mean_all(&l2_vec(pred, target)).0 as f32
}

/// Provides the vector derivative of the mean squared error
pub fn mse_derivative(pred: &Array<f32>, target: &Array<f32>) -> Array<f32> {
    af::sub(pred, target, false)
}

/// Helper to provide a loss from a string
pub fn get_loss(name: &str, pred: &Array<f32>, target: &Array<f32>) -> Result<f32, HALError> {
    match name {
        "mse" => Ok(mse(pred, target)),
        _ => Err(HALError::UNKNOWN),
    }
}

/// Helper to provide a loss derivative from a string
pub fn get_loss_derivative(
    name: &str,
    pred: &Array<f32>,
    target: &Array<f32>,
) -> Result<Array<f32>, HALError> {
    match name {
        "mse" => Ok(mse_derivative(pred, target)),
        _ => Err(HALError::UNKNOWN),
    }
}
