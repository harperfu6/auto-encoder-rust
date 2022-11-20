mod sequential;
use std::collections::HashMap;

pub use self::sequential::Sequential;
use crate::data::DataSouce;
use crate::optimizer::Optimizer;

use arrayfire::Array;

pub trait Model {
    fn new(optimizer: Box<dyn Optimizer>, loss: &str) -> Self;

    /// Adds a new layer to the sequential model
    ///
    /// Given a layer type and provided parameters this function
    /// will add the required parameters to the sequential model
    ///
    /// # Parameters
    ///
    /// - `layer` is the type of layer to add
    /// - `params` is a hashmap of params for the provided layer
    fn add(&mut self, layer: &str, params: HashMap<&str, String>);

    /// Fit's model to provided data
    ///
    /// Given input and output data, fits the model the given data by running
    /// both forward and backward pass on the data and optimizing the system per minibatch
    ///
    /// # Parameters
    ///
    /// - `source` is the datasource
    /// - `src_device` is the source device of the data
    /// - `epochs` is the number of epochs to run the training loop for
    /// - `batch_size` is the minibatch size
    /// - `bptt_interval` is the optional parameter for truncated backprop through time (RNN's only)
    /// - `loss_indices` are the indices to utilize when doing backward pass (useful for RNN long term tasks)
    /// - `verbose` specifies whether or not to print verbose details during training
    ///
    /// # Return Values
    ///
    /// Vector of losses
    fn fit<T>(
        &mut self,
        source: &T,
        epochs: u64,
        batch_size: u64,
        loss_indices: Option<&Vec<bool>>,
        verbose: bool,
    ) -> Vec<f32>
    where
        T: DataSouce;

    /// Calculate the forward pass of all the layers
    ///
    /// Given an array of inputs this function computes the forward pass
    /// on all the available layers and return the final model outputs
    ///
    /// # Parameters
    ///
    /// - `inputs` is an array of activations [batch, feature, time]
    /// - `src_device` is the source device that the data is coming from
    /// - `dest_device` is the destination device that the data should go to
    ///
    /// # Return Values
    ///
    /// Vector of activated outputs of the model
    fn forward(&self, inputs: &Array<f32>) -> Vec<Array<f32>>;

    /// Calculate the layer gradients and return the loss vector
    ///
    /// Given predictions and output data, this function computes all the gradients for all
    /// of the trainable parameters in the layers
    ///
    /// # Parameters
    ///
    /// - `predictions` are the model predictions
    /// - `targets` are the true targets
    /// - `loss_indices` are the optional indices of losses to use while computing the gradient
    ///
    /// # Return Values
    ///
    /// Vector of losses
    fn backward(
        &mut self,
        predictions: &Vec<Array<f32>>,
        targets: &Array<f32>,
        loss_indices: Option<&Vec<bool>>,
    ) -> Vec<f32>;

    /// Show model info
    ///
    ///
    fn info(&self);
}
