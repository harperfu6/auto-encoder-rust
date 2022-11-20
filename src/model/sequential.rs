use std::collections::HashMap;

use crate::layer::{Dense, Layer};
use crate::loss;
use crate::model::Model;
use crate::optimizer::{Optimizer, SGD};
use crate::params::{DenseGenerator, ParamManager};
use crate::utils;

use af::print;
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

    fn fit<T>(
        &mut self,
        source: &T,
        epochs: u64,
        batch_size: u64,
        loss_indices: Option<&Vec<bool>>,
        verbose: bool,
    ) -> Vec<f32>
    where
        T: crate::data::DataSouce,
    {
        // some simple data validation check
        let data_params = source.info();
        let idims = data_params.input_dims;
        let tdims = data_params.target_dims;
        let iters = data_params.num_samples as u64 / batch_size as u64;
        println!(
            "\ntrain samples: {:?} | target samples: {:?} | batch size: {}",
            idims, tdims, batch_size
        );
        println!("epochs: {} | iterations[per epoch]: {}", epochs, iters);
        assert!(
            idims[0] == tdims[0],
            "batch sizes for inputs and targets much be equal"
        );
        assert!(
            idims[2] == tdims[2],
            "sequence lengths for inputs and targets must be equal"
        );
        assert!(self.layers.len() > 0, "Need at least one layer to fit!");

        let mut lossvec = Vec::<f32>::new();
        for epoch in 0..epochs {
            for iter in 0..iters {
                if verbose {
                    print!("\n[epoch: {}][iter: {}]", epoch, iter);
                }
                let minibatch = source.get_train_iter(batch_size);
                assert!(
                    minibatch.input.dims()[0] == batch_size,
                    "Ensure that input dims are of batch rows"
                );
                assert!(
                    minibatch.target.dims()[0] == batch_size,
                    "Ensure that target dims are of batch rows"
                );

                let batch_input = &minibatch.input;
                let batch_target = &minibatch.target;

                let mut current_loss_vec = Vec::new();
                let a_t = self.forward(&batch_input);
                current_loss_vec = self.backward(&a_t, &batch_target, loss_indices);

                self.optimizer
                    .update(&mut self.param_manager, batch_size as u64);

                if verbose {
                    let loss_sum = current_loss_vec.iter().fold(0f32, |sum, val| sum + val);
                    let avg_loss = loss_sum / current_loss_vec.len() as f32;
                    print!("{} ", avg_loss);
                }

                lossvec.extend(current_loss_vec);
            }
        }

        lossvec
    }

    fn forward(&self, inputs: &Array<f32>) -> Vec<Array<f32>> {
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
        targets: &Array<f32>,
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
