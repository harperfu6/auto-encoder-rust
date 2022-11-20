use af::{Array, Dim4, HasAfEnum};

use crate::utils;

pub fn plot_array(values: &Array<f32>, title: &str, window_x: u16, window_y: u16) {
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

pub fn plot_vec(raw_values: Vec<f32>, title: &str, window_x: u16, window_y: u16) {
    let num_rows = raw_values.len();
    let dims = Dim4::new(&[num_rows as u64, 1, 1, 1]);
    let values = utils::vec_to_array(raw_values, dims);
    plot_array(&values, title, window_x, window_y);
}

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
    // fn basic_sin_wave() {
    //     let x_vec: Vec<f32> = dbg!((0..600).map(|x| -3.14 + ((x as f32) * 0.01)).collect());
    //     let x_arr: Array<f32> = Array::new(&x_vec, Dim4::new(&[600, 1, 1, 1]));
    //     let y_arr: Array<f32> = af::sin(&x_arr);
    //     plot_array(&y_arr, "sin", 1280, 1280);
    // }

    // #[test]
    // fn generated_sin_wave() {
    //     let input_dims = 2;
    //     let num_rows = 50;
    //     let sin_source = SinSource::new(DType::F32);
    //     let arr = sin_source.generate_sin_wave(input_dims, num_rows);
    //     print(&arr);
    //     // plot_array(&arr, "generated sin", 1280, 1280);
    // }
}
