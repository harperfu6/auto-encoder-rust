use af::Array;
use arrayfire::Dim4;

use crate::initializations;

#[derive(Debug)]
pub struct Params {
    pub weights: Vec<Array<f32>>,
    pub biases: Vec<Array<f32>>,
    pub deltas: Vec<Array<f32>>,
    pub activations: Vec<String>,
}

pub struct ParamManager {
    pub layer_storage: Vec<Params>,
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

        self.layer_storage.push(Params {
            weights,
            biases,
            deltas,
            activations: owned_activations,
        });
    }

    fn generate(&self, init: &str, dims: (usize, usize)) -> Array<f32> {
        let dims = Dim4::new(&[dims.0 as u64, dims.1 as u64, 1, 1]);
        initializations::get_initialization(init, dims).unwrap()
    }
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
