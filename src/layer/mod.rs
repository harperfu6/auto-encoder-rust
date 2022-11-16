mod dense;
use crate::params::Params;
use crate::{activations, params};
use arrayfire::{Array, MatProp};

pub trait Layer {
    fn forward(&self, params: Params, inputs: &Array<f32>) -> Array<f32>;
    fn backward(&self, params: Params, delta: &Array<f32>) -> Array<f32>;
}

/// Helper to run f(wx + b) where bias is optional
pub fn linear(
    input: &Array<f32>,
    weight: &Array<f32>,
    bias: Option<&Array<f32>>,
    activation: &str,
) -> Array<f32> {
    // w_x = xW
    // z_t = w_x + b
    let xw = af::matmul(input, weight, MatProp::NONE, MatProp::NONE);
    let z_t = match bias {
        Some(b) => af::transpose(&af::add(&af::transpose(&xw, false), b, true), false),
        None => xw.clone(),
    };

    activations::get_activation(activation, &z_t).unwrap()
}

/// Helper that computes the backward operation on f(wx + b) and returns delta, dW, db
pub fn linear_backward(
    delta: &Array<f32>,
    input: &Array<f32>,
    output: &Array<f32>,
    activation: &str,
) -> (Array<f32>, Array<f32>, Array<f32>) {
    // delta_t = (transpose(W_{t+1}) * d_{l+1} .* dActivation(z))
    // delta_{t-1} = (transpose(W_t) * d_{l})
    let dz = activations::get_derivative(activation, output).unwrap();
    let delta_t = af::mul(delta, &dz, false);
    let dw = af::matmul(input, &delta_t, MatProp::TRANS, MatProp::NONE);
    let db = af::transpose(&af::sum(&delta_t, 0), false);

    return (delta_t, dw, db);
}
