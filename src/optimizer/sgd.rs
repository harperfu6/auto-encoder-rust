use std::collections::HashMap;

use af;
use arrayfire::Array;
use itertools::multizip;

use crate::{initializations, optimizer::Optimizer};

pub struct SGD {
    name: String,
    velocity: Vec<Array<f32>>,
    iter: u64,
    learning_rate: f32,
    momemtum: f32,
    decay: f32,
}

impl Default for SGD {
    fn default() -> Self {
        SGD {
            name: "SGD".to_string(),
            velocity: Vec::new(),
            iter: 0,
            learning_rate: 1e-3,
            momemtum: 0.0,
            decay: 0.0,
        }
    }
}

impl Optimizer for SGD {
    fn new(params: &HashMap<&str, &str>) -> SGD {
        SGD {
            name: "SGD".to_string(),
            velocity: Vec::new(),
            iter: 0,
            learning_rate: params.get("learning_rate").unwrap().parse::<f32>().unwrap(),
            momemtum: params.get("momemtum").unwrap().parse::<f32>().unwrap(),
            decay: params.get("decay").unwrap().parse::<f32>().unwrap(),
        }
    }

    fn setup(&mut self, dims: Vec<arrayfire::Dim4>) {
        if self.velocity.len() == 0 {
            for dim in dims {
                self.velocity.push(initializations::zeros(dim));
            }
        }
    }

    fn update(&mut self, parameter_manager: &mut crate::params::ParamManager, batch_size: u64) {
        self.iter += 1;
        let lr = self.learning_rate * (1.0 / (1.0 + self.decay * (self.iter as f32)));
        let alpha = lr / batch_size as f32;

        // all arrays are returned as [W0, b0, .. WN, bN, ..] (note this is per layer)
        // deltas are returned in the same way
        let num_params = self.velocity.len();
        for (arr, delta, velocity, ind) in multizip((
            parameter_manager.get_all_arrays().iter(),
            parameter_manager.get_all_deltas().iter(),
            self.velocity.iter_mut(), // velocityは更新する前提のため
            0..num_params,
        )) {
            let grad_update = delta.clone();

            // v   = momemtum * v + learning_rate * d_w (or d_b)
            // p   = p - v
            *velocity = af::add(
                &af::mul(&self.momemtum, velocity, false),
                &af::mul(&alpha, &grad_update, false),
                false,
            );
            assert!(velocity.dims().get() == arr.dims().get());
            parameter_manager.set_array_from_index(af::sub(arr, velocity, false), ind);
        }

        parameter_manager.zero_all_deltas();
    }
}
