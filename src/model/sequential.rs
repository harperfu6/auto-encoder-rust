use std::collections::HashMap;

use crate::layer::Layer;
use crate::model::Model;
use crate::optimizer::{Optimizer, SGD};
use crate::params::{DenseGenerator, ParamManager};

use arrayfire::Array;

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    param_manager: ParamManager,
    optimizer: Box<dyn Optimizer>,
    loss: String,
}

impl Default for Sequential {
    fn default() -> Sequential {
        Sequential {
            layers: Vec::new(),
            param_manager: ParamManager::default(),
            optimizer: Box::new(SGD::default()),
            loss: "mse".to_string(),
        }
    }
}

impl Model for Sequential {
    fn new(optimizer: Box<dyn Optimizer>, loss: &str) -> Sequential {
        Sequential {
            layers: Vec::new(),
            param_manager: ParamManager::default(),
            optimizer,
            loss: loss.to_string(),
        }
    }

    fn add(&mut self, layer: &str, params: HashMap<&str, String>) {
        let input_size = params.get("input_size").unwrap().parse::<u64>().unwrap() as usize;
        let output_size = params.get("output_size").unwrap().parse::<u64>().unwrap() as usize;
        let activation = params.get("activation").unwrap();
        let w_init = params.get("w_init").unwrap();
        let b_init = params.get("b_init").unwrap();

        match layer {
            "dense" => {
                self.param_manager
                    .add_dense(input_size, output_size, activation, w_init, b_init);
            }
            _ => {
                panic!("not supported layer!!");
            }
        }
    }

    fn fit<T>(self, source: &T, epochs: u64, batch_size: u64) -> Vec<f32>
    where
        T: crate::data::DataSouce,
    {
        todo!()
    }

    fn forward(self, inputs: &Array<f32>) -> Vec<Array<f32>> {
        let mut outputs: Vec<Array<f32>> = Vec::new();
        for i in 0..self.layers.len() {
            let array = self.layers[i].forward(inputs);
            outputs.push(array)
        }
        outputs
    }

    fn backward(self, predictions: &Vec<Array<f32>>, targets: Array<f32>) {
        todo!()
    }

    fn info(&self) {
        println!("model info!!");
        dbg!(&self.param_manager.layer_storage);
        // println!("num_layers: {}", self.layers.len());
    }
}
