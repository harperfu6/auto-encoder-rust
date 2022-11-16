use af::DType;

mod sin;

pub struct Data {}

struct DataParams {
    dtypes: DType,
}

pub trait DataSouce {
    fn get_train_iter(&self, num_batch: u64) -> Data;
    fn get_test_iter(&self, num_batch: u64) -> Data;
}
