use af::{print, Array, DType, Dim4, HasAfEnum};
use std::cell::Cell;

use crate::data::{Data, DataParams, DataSouce};

pub struct SinSource {
    pub params: DataParams,
    pub offset: Cell<f32>,
}

impl SinSource {
    pub fn new(input_size: u64, batch_size: u64, dtype: DType, max_samples: u64) -> SinSource {
        let dims = Dim4::new(&[batch_size, input_size, 1, 1]);
        SinSource {
            params: DataParams {
                input_dims: dims,
                target_dims: dims,
                dtypes: dtype,
                num_samples: max_samples,
            },
            offset: Cell::new(0.0f32),
        }
    }

    fn generate_sin_wave(&self, input_dims: u64, num_rows: u64) -> Array<f32> {
        let tdims = Dim4::new(&[input_dims, num_rows, 1, 1]);
        let dims = Dim4::new(&[1, num_rows * input_dims, 1, 1]);
        let x = af::transpose(&af::moddims(&af::range::<f32>(dims, 1), tdims), false);

        let x_shifted = af::add(
            &self.offset.get(),
            &af::div(&x, &(input_dims), false),
            false,
        );
        self.offset
            .set(self.offset.get() + 1.0 / (input_dims * num_rows - 1) as f32);

        print(&x_shifted);
        af::sin(&x_shifted)
        // utils::cast(&af::sin(&x_shifted), self.params.dtype)
    }
}

impl DataSouce for SinSource {
    fn info(&self) -> DataParams {
        self.params.clone()
    }

    fn get_train_iter(&self, num_batch: u64) -> Data {
        let inp = self.generate_sin_wave(self.params.input_dims[1], num_batch);
        let batch = Data {
            input: inp.clone(),
            target: inp.clone(),
        };
        batch
    }

    fn get_test_iter(&self, num_batch: u64) -> Data {
        self.get_train_iter(num_batch)
    }
}
