use std::sync::{Arc, Mutex};

use crate::layer;
use crate::layer::Layer;
use crate::params::Params;
use af::{Array, MatProp};

pub struct Dense {
    pub input_size: usize,
    pub output_size: usize,
}

impl Layer for Dense {
    fn forward(&self, params: Arc<Mutex<Params>>, inputs: &Array<f32>) -> Array<f32> {
        let mut ltex = params.lock().unwrap();

        // output is simple linear operation
        // denseの1forward分は1層分しかないので先頭分のみを取得?
        let a_t = layer::linear(
            inputs,
            &ltex.weights[0],
            Some(&ltex.biases[0]),
            &ltex.activations[0],
        );

        // save inputs/outputs
        ltex.inputs.push(inputs.clone());
        ltex.outputs.push(a_t.clone());

        a_t
    }

    fn backward(&self, params: Arc<Mutex<Params>>, delta: &Array<f32>) -> Array<f32> {
        let mut ltex = params.lock().unwrap();
        // utilize the helper to get our deltas
        let (delta_t, dw, db) = layer::linear_backward(
            delta,
            &ltex.inputs[0],
            &ltex.outputs[0],
            &ltex.activations[0],
        );

        ltex.deltas[0] = af::add(&ltex.deltas[0], &dw, false);
        ltex.deltas[1] = af::add(&ltex.deltas[1], &db, false);

        af::matmul(&delta_t, &ltex.weights[0], MatProp::NONE, MatProp::TRANS)
    }
}
