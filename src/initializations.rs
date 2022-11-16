use crate::{error::HALError, utils};
use af::{Array, Dim4, HasAfEnum};
use arrayfire::FloatingPoint;
use rand::{self, Rng};

/// A helper to return a normal shape
pub fn normal<T: HasAfEnum + FloatingPoint>(dims: Dim4) -> Array<T> {
    let mut rng = rand::thread_rng();
    af::set_seed(rng.gen::<u64>());

    let src_type = T::get_af_dtype();
    let u = af::randn::<T>(dims);
    let dst_type = u.get_type();
    assert!(
        src_type == dst_type,
        "type mismatch detected in normal, {:?} vs {:?}",
        src_type,
        dst_type
    );
    u
}

/// A helper to provide a uniform shape
pub fn uniform<T: HasAfEnum + FloatingPoint>(dims: Dim4) -> Array<T> {
    let mut rng = rand::thread_rng();
    af::set_seed(rng.gen::<u64>());

    let src_type = T::get_af_dtype();
    let u = af::randu::<T>(dims);
    let dst_type = u.get_type();
    assert!(
        src_type == dst_type,
        "type mismatch detected in normal, {:?} vs {:?}",
        src_type,
        dst_type
    );
    u
}

/// A helper to provide a shape of zeros
pub fn zeros(dims: Dim4) -> Array<f32> {
    utils::constant(dims, 0.0f32)
}

/// A helper to provide a shape of ones
pub fn ones(dims: Dim4) -> Array<f32> {
    utils::constant(dims, 1.0f32)
}

pub fn get_initialization(name: &str, dims: Dim4) -> Result<Array<f32>, HALError> {
    match name {
        "normal" => Ok(normal(dims)),
        "uniform" => Ok(uniform(dims)),
        "zeros" => Ok(zeros(dims)),
        "ones" => Ok(ones(dims)),
        _ => Err(HALError::UNKNOWN),
    }
}
