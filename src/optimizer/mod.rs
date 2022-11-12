mod sgd;
pub use self::sgd::SGD;

pub trait Optimizer {
    fn new() -> Self
    where
        Self: Sized;
}
