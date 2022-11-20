use std::sync::{Arc, Mutex};

use af::Array;
use arrayfire::Dim4;

use crate::{initializations, utils};

macro_rules! check_layer_index_overflow {
    ($self: ident, $layer_index: ident) => {
        assert!(&$self.layer_storage.len() - 1 >= $layer_index);
    };
}

macro_rules! set_param_func {
    ($fn_name: ident, $vec_extension: ident, $base_type: ty) => {
        pub fn $fn_name(&self, layer_index: usize, num: usize, p: $base_type) {
            assert!(self.layer_storage.len() - 1 >= layer_index);
            let layer = self.layer_storage[layer_index].clone();
            let mut ltex = layer.lock().unwrap();
            let ext = &mut ltex.$vec_extension;
            ext[num] = p;
        }
    };
}

macro_rules! get_param_vec_func {
    ($fn_name: ident, $vec_extension: ident, $base_type: ty) => {
        pub fn $fn_name(&self, layer_index: usize) -> Vec<$base_type> {
            assert!(self.layer_storage.len() - 1 >= layer_index);
            let layer = self.layer_storage[layer_index].clone();
            let ltex = layer.lock().unwrap();
            ltex.$vec_extension.clone()
        }
    };
}

macro_rules! get_param_func {
    ($fn_name: ident, $vec_extension: ident, $base_type: ty) => {
        pub fn $fn_name(&self, layer_index: usize, num: usize) -> $base_type {
            assert!(self.layer_storage.len() - 1 >= layer_index);
            let layer = self.layer_storage[layer_index].clone();
            let ltex = layer.lock().unwrap();
            let ext = &ltex.$vec_extension;
            assert!(ext.len() - 1 >= num);
            ext[num].clone()
        }
    };
}

#[derive(Debug, Clone)]
pub struct Params {
    pub weights: Vec<Array<f32>>,
    pub biases: Vec<Array<f32>>,
    pub deltas: Vec<Array<f32>>,
    pub activations: Vec<String>,
    pub inputs: Vec<Array<f32>>,
    pub outputs: Vec<Array<f32>>,
    pub current_unroll: usize,
}

pub struct ParamManager {
    pub layer_storage: Vec<Arc<Mutex<Params>>>,
}

impl Default for ParamManager {
    fn default() -> Self {
        ParamManager {
            layer_storage: Vec::new(),
        }
    }
}

impl ParamManager {
    fn add(
        &mut self,
        layer_type: &str,
        weight_params: Vec<(&str, (usize, usize))>, // (init, (input, output))
        bias_params: Vec<(&str, (usize, usize))>,   // (init, (input, output)
        activations: Vec<&str>,
    ) {
        let num_params = weight_params.len() + bias_params.len();
        let mut deltas: Vec<Array<f32>> = Vec::with_capacity(num_params); // for each w/b

        // generate the weights
        let mut weights: Vec<Array<f32>> = Vec::with_capacity(weight_params.len());
        for (w_init, w_dims) in weight_params {
            weights.push(self.generate(w_init, w_dims));
            deltas.push(self.generate("zeros", w_dims));
        }

        // generate the biases
        let mut biases: Vec<Array<f32>> = Vec::with_capacity(bias_params.len());
        for (b_init, b_dims) in bias_params {
            biases.push(self.generate(b_init, b_dims));
            deltas.push(self.generate("zeros", b_dims));
        }

        // activations
        let owned_activations = activations
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>();

        self.layer_storage.push(Arc::new(Mutex::new(Params {
            weights,
            biases,
            deltas,
            activations: owned_activations,
            inputs: Vec::new(),
            outputs: Vec::new(),
            current_unroll: 0,
        })));
    }

    fn generate(&self, init: &str, dims: (usize, usize)) -> Array<f32> {
        let dims = Dim4::new(&[dims.0 as u64, dims.1 as u64, 1, 1]);
        initializations::get_initialization(init, dims).unwrap()
    }

    pub fn num_layers(&self) -> usize {
        self.layer_storage.len()
    }

    pub fn num_weights(&self, layer_index: usize) -> usize {
        check_layer_index_overflow!(self, layer_index);
        let layer = self.layer_storage[layer_index].clone();
        let ltex = layer.lock().unwrap();
        ltex.weights.len()
    }

    pub fn num_biases(&self, layer_index: usize) -> usize {
        check_layer_index_overflow!(self, layer_index);
        let layer = self.layer_storage[layer_index].clone();
        let ltex = layer.lock().unwrap();
        ltex.biases.len()
    }

    pub fn num_arrays(&self, layer_index: usize) -> usize {
        check_layer_index_overflow!(self, layer_index);
        self.num_biases(layer_index) + self.num_weights(layer_index)
    }

    // pub fn num_state_derivatives(&self, layer_index: usize) {
    //     check_layer_index_overflow!(self, layer_index);
    //     let layer = self.layer_storage[layer_index].clone();
    //     let ltex = layer.lock().unwrap();
    //     ltex.
    // }

    pub fn get_params(&self, layer_index: usize) -> Arc<Mutex<Params>> {
        check_layer_index_overflow!(self, layer_index);
        self.layer_storage[layer_index].clone()
    }

    pub fn get_all_arrays(&self) -> Vec<Array<f32>> {
        let mut p = Vec::new();
        for layer_num in 0..self.num_layers() {
            p.extend(self.get_weights(layer_num));
            p.extend(self.get_biases(layer_num));
        }
        p
    }

    pub fn get_all_deltas(&self) -> Vec<Array<f32>> {
        let mut p = Vec::new();
        for layer_num in 0..self.num_layers() {
            p.extend(self.get_deltas(layer_num));
        }
        p
    }

    pub fn get_all_dims(&self) -> Vec<Dim4> {
        let mut dims = Vec::new();
        for layer in &self.layer_storage {
            let ltex = layer.lock().unwrap();
            for w in &ltex.weights {
                dims.push(w.dims().clone());
            }
            for b in &ltex.biases {
                dims.push(b.dims().clone());
            }
        }
        dims
    }

    // assumes params are coming in layer wise
    // eg: [W0, b0, .. , WN, bN]
    pub fn set_array_from_index(&self, arr: Array<f32>, ind: usize) {
        let mut current: usize = 0;
        for layer_num in 0..self.num_layers() {
            let n_weights = self.num_weights(layer_num);
            let num_biases = self.num_biases(layer_num);

            if current + n_weights > ind {
                // we are a weights
                let w_index = ind - current;
                let target_dims = self.get_weight(layer_num, w_index).dims();
                let src_dims = arr.dims();
                self.set_weight(layer_num, w_index, arr);
                break;
            }

            current += n_weights;
            if current + num_biases > ind {
                // we are a bias
                let b_index = ind - current;
                self.set_bias(layer_num, b_index, arr);
                break;
            }
            current += num_biases;
        }
    }

    pub fn zero_all_deltas(&self) {
        for layer_num in 0..self.num_layers() {
            for delta_num in 0..self.num_arrays(layer_num) {
                let delta = self.get_delta(layer_num, delta_num);
                let delta_dims = delta.dims();
                let zero_tensor = utils::constant(delta_dims, 0.0f32);
                self.set_delta(layer_num, delta_num, zero_tensor);
            }
        }
    }

    // pub fn zero_all_state_derivatives(&self) {
    //     for layer_num in 0..self.num_layers() {
    //         for state_num in
    //     }
    // }

    get_param_func!(get_weight, weights, Array<f32>);
    get_param_func!(get_delta, deltas, Array<f32>);

    get_param_vec_func!(get_outputs, outputs, Array<f32>);
    get_param_vec_func!(get_weights, weights, Array<f32>);
    get_param_vec_func!(get_biases, biases, Array<f32>);
    get_param_vec_func!(get_deltas, deltas, Array<f32>);

    set_param_func!(set_weight, weights, Array<f32>);
    set_param_func!(set_bias, biases, Array<f32>);
    set_param_func!(set_delta, deltas, Array<f32>);
}

/** Custom Layer Trait **/
pub trait DenseGenerator {
    fn add_dense(
        &mut self,
        input_size: usize,
        output_size: usize,
        activation: &str,
        w_init: &str,
        b_init: &str,
    );
}

impl DenseGenerator for ParamManager {
    fn add_dense(
        &mut self,
        input_size: usize,
        output_size: usize,
        activation: &str,
        w_init: &str,
        b_init: &str,
    ) {
        self.add(
            "dense",
            vec![(w_init, (input_size, output_size))],
            vec![(b_init, (output_size, 1))],
            vec![activation],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro() {
        let mut pm = ParamManager::default();
        pm.add(
            "dense",
            vec![("normal", (2, 2))],
            vec![("zeros", (2, 1))],
            vec!["tanh"],
        );
        dbg!(pm.get_outputs(0));
    }
}
