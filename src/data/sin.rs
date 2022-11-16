use std::cell::Cell;

use af::{print, Array, DType, Dim4};
use arrayfire::HasAfEnum;

use crate::data::{Data, DataParams, DataSouce};

fn plot_array(values: &Array<f32>, title: &str, window_x: u16, window_y: u16) {
    assert!(values.dims()[1] == 1);

    let wnd = af::Window::new(window_x as i32, window_y as i32, title.to_string());

    // display till closed
    loop {
        wnd.draw_plot2(&af::range::<f32>(values.dims(), 0), &values, None);
        if wnd.is_closed() == true {
            break;
        }
    }
}

struct SinSource {
    params: DataParams,
    offset: Cell<f32>,
}

impl SinSource {
    fn new(dtype: DType) -> SinSource {
        SinSource {
            params: DataParams { dtypes: dtype },
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

// impl DataSouce for SinSource {
//     fn get_train_iter(&self, num_batch: u64) -> Data {}

//     fn get_test_iter(&self, num_batch: u64) -> Data {}
// }

#[cfg(test)]
mod tests {
    use super::*;
    use af::print;

    // #[test]
    // fn print_array() {
    //     let values: [f32; 3] = [1.0, 2.0, 3.0];
    //     let indices = Array::new(&values, Dim4::new(&[3, 1, 1, 1]));
    //     print(&indices);
    // }

    // #[test]
    // fn simple_sin_wage() {
    //     let x_vec: Vec<f32> = dbg!((0..600).map(|x| -3.14 + ((x as f32) * 0.01)).collect());
    //     let x_arr: Array<f32> = Array::new(&x_vec, Dim4::new(&[600, 1, 1, 1]));
    //     let y_arr: Array<f32> = af::sin(&x_arr);
    //     plot_array(&y_arr, "sin", 1280, 1280);
    // }

    // #[test]
    // fn generate_sin_wave() {
    //     let input_dims = 2;
    //     let num_rows = 50;
    //     let sin_source = SinSource::new(DType::F32);
    //     let arr = sin_source.generate_sin_wave(input_dims, num_rows);
    //     print(&arr);
    //     // plot_array(&arr, "generated sin", 1280, 1280);
    // }
}
