use af::DType;

mod sin;

struct Data {}

struct DataParams {
    dtypes: DType,
}

trait DataSouce {
    fn get_train_iter(&self, num_batch: u64) -> Data;
    fn get_test_iter(&self, num_batch: u64) -> Data;
}
