use crate::layer;
use crate::layer::Layer;
use crate::params::Params;
use af::Array;

pub struct Dense {
    input_size: usize,
    output_size: usize,
}

impl Layer for Dense {
    fn forward(&self, params: Params, inputs: &Array<f32>) -> Array<f32> {
        // output is simple linear operation
        // denseの各種パラメータは一層分しかないので先頭を取得
        let a_t = layer::linear(
            inputs,
            &params.weights[0],
            Some(&params.biases[0]),
            &params.activations[0],
        );

        a_t
    }

    fn backward(&self, params: Params, delta: &Array<f32>) -> Array<f32> {
        // utilize the helper to get our deltas
        let (delta_t, dw, db) = layer::linear_backward(delta, input, output, activation) 
    }
}
