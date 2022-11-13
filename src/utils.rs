use af::{Array, DType, HasAfEnum};
use num::Complex;

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
