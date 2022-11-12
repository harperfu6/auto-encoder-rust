use crate::layer::Layer;
use crate::model::Model;
use crate::optimizer::{Optimizer, SGD};

use arrayfire::Array;

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Box<dyn Optimizer>,
    loss: String,
}

impl Default for Sequential {
    fn default() -> Sequential {
        Sequential {
            layers: Vec::new(),
            optimizer: Box::new(SGD::default()),
            loss: "mse".to_string(),
        }
    }
}

impl Model for Sequential {
    fn new(optimizer: Box<dyn Optimizer>, loss: &str) -> Sequential {
        Sequential {
            layers: Vec::new(),
            optimizer,
            loss: loss.to_string(),
        }
    }

    fn forward(self, inputs: &Array<f32>) -> Vec<Array<f32>> {
        let mut outputs: Vec<Array<f32>> = Vec::new();
        for i in 0..self.layers.len() {
            let array = self.layers[i].forward(inputs);
            outputs.push(array)
        }
        outputs
    }

    fn add(layer: &str) {
        match layer {
            "dense" => {}
            _ => {}
        }
    }
}
