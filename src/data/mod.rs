use af::{Array, DType, Dim4};

pub use self::sin::SinSource;
mod sin;

pub struct Data {
    pub input: Array<f32>,
    pub target: Array<f32>,
}

#[derive(Clone)]
pub struct DataParams {
    pub input_dims: Dim4,
    pub target_dims: Dim4,
    pub dtypes: DType,
    pub num_samples: u64,
}

/// A DataSource needs to provide these basic features
///
/// 1) It gives information regarding the source
/// 2) It provides a train iterator that returns a minibatch
/// 3) It provides a test iterator that returns a minibatch
/// 4) It (optionally)provides a validation iterator that returns a minibatch
pub trait DataSouce {
    fn info(&self) -> DataParams;
    fn get_train_iter(&self, num_batch: u64) -> Data;
    fn get_test_iter(&self, num_batch: u64) -> Data;
}
