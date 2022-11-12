use arrayfire::Array;

pub trait Layer {
    fn forward(&self, inputs: &Array<f32>) -> Array<f32>;
}
