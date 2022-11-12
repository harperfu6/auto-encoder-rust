mod sequential;
pub use self::sequential::Sequential;
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
    fn add(layer: &str);

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
    fn forward(self, inputs: &Array<f32>) -> Vec<Array<f32>>;
}
