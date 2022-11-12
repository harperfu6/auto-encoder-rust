use crate::optimizer::Optimizer;

pub struct SGD {
    name: String,
}

impl Default for SGD {
    fn default() -> Self {
        SGD {
            name: "SGD".to_string(),
        }
    }
}

impl Optimizer for SGD {
    fn new() -> SGD {
        SGD {
            name: "SGD".to_string(),
        }
    }
}
