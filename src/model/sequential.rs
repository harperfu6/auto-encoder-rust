use std::collections::HashMap;

use crate::layer::{Dense, Layer};
use crate::loss;
use crate::model::Model;
use crate::optimizer::{Optimizer, SGD};
use crate::params::{DenseGenerator, ParamManager};
use crate::utils;

use arrayfire::Array;
use itertools::multizip;

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
                self.layers.push(Box::new(Dense {
                    input_size,
                    output_size,
                }));
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
        let mut activate = inputs.clone();

        for i in 0..self.layers.len() {
            let a = self.layers[i].forward(self.param_manager.get_params(i), &activate);
            activate = a;
        }

        // return the collected outputs of the last layer
        let last_index = self.layers.len() - 1;
        let outputs: Vec<Array<f32>> = self.param_manager.get_outputs(last_index);

        outputs
    }

    fn backward(
        &mut self,
        predictions: &Vec<Array<f32>>,
        targets: Array<f32>,
        loss_indices: Option<&Vec<bool>>,
    ) -> Vec<f32> {
        self.optimizer.setup(self.param_manager.get_all_dims());
        let mut loss_vec = Vec::with_capacity(predictions.len());

        for (pred, ind) in multizip((predictions.iter().rev(), (0..predictions.len()).rev())) {
            let tar = af::slice(&targets, ind as i64);
            let last_index = self.layers.len();

            let mut delta = match loss_indices {
                Some(li) => match li[ind] {
                    false => utils::constant(tar.dims(), 0.0f32),
                    true => {
                        loss_vec.push(loss::get_loss(&self.loss, pred, &tar).unwrap());
                        loss::get_loss_derivative(&self.loss, pred, &tar).unwrap()
                    }
                },
                None => {
                    loss_vec.push(loss::get_loss(&self.loss, pred, &tar).unwrap());
                    loss::get_loss_derivative(&self.loss, pred, &tar).unwrap()
                }
            };

            for i in (0..last_index).rev() {
                delta = self.layers[i].backward(self.param_manager.get_params(i), &delta);
            }
        }

        loss_vec
    }

    fn info(&self) {
        println!("model info!!");
        dbg!(&self.param_manager.layer_storage);
        // println!("num_layers: {}", self.layers.len());
    }
}
