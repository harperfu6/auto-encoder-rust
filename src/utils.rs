use af::{Array, DType, Dim4, HasAfEnum};
use num::Complex;

#[macro_export]
macro_rules! hashmap {
    ($( $key: expr => $val: expr ), *) => {{
        let mut map = std::collections::HashMap::new();
        $( map.insert($key, $val); )*
        map
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macro_hashmap() {
        let input_dims = 128;
        let output_dims = 128;
        let hm = hashmap!(
        "activation" => "tanh".to_string(),
        "input_size" => input_dims.to_string(),
        "output_size" => output_dims.to_string()
        );
        dbg!(hm);
    }
}

/// A Helper to return a constant value based on type
// pub fn constant(dims: Dim4, aftype: DType, val: f32) -> Array<f32> {
pub fn constant(dims: Dim4, val: f32) -> Array<f32> {
    af::constant(val, dims)
    // match aftype {
    //     DType::F32 => af::constant(val, dims),
    //     DType::F64 => af::constant(val.approx_as::<f64>().unwrap(), dims),
    //     DType::C32 => af::constant(Complex::new(val, 0f32), dims),
    //     DType::C64 => af::constant(Complex::new(val.approx_as::<f64>().unwrap(), 0f64), dims),
    //     DType::B8 => {
    //         if val > 0f32 {
    //             af::constant(true, dims)
    //         } else {
    //             af::constant(false, dims)
    //         }
    //     }
    //     DType::S32 => af::constant(val.approx_as::<i32>().saturate().unwrap(), dims),
    //     DType::U32 => af::constant(val.approx_as::<u32>().saturate().unwrap(), dims),
    //     DType::U8 => af::constant(val.approx_as::<u8>().saturate().unwrap(), dims),
    //     DType::S64 => af::constant(val.approx_as::<i64>().saturate().unwrap(), dims),
    //     DType::U64 => af::constant(val.approx_as::<u64>().saturate().unwrap(), dims),
    //     DType::S16 => af::constant(val.approx_as::<i16>().saturate().unwrap(), dims),
    //     DType::U16 => af::constant(val.approx_as::<u16>().saturate().unwrap(), dims),
    //     _ => af::constant(val, dims),
    // }
}

// pub fn cast<T: HasAfEnum>(input: &Array<T>, dest_type: DType) -> Array<T> {
//     if input.get_type() == dest_type {
//         return input.clone();
//     }

//     match dest_type {
//         DType::F32 => input.cast::<f32>(),
//         DType::F64 => input.cast::<f64>(),
//         DType::C32 => input.cast::<Complex<f32>>(),
//         DType::C64 => input.cast::<Complex<f64>>(),
//         DType::B8 => input.cast::<bool>(),
//         DType::S32 => input.cast::<i32>(),
//         DType::U32 => input.cast::<u32>(),
//         DType::U8 => input.cast::<u8>(),
//         DType::S64 => input.cast::<i64>(),
//         DType::U64 => input.cast::<u64>(),
//         DType::S16 => input.cast::<i64>(),
//         DType::U16 => input.cast::<u16>(),
//         _ => input.cast::<f32>(),
//     }
// }
