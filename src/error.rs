#[derive(Debug)]
pub enum HALError {
    ///
    /// The function returned successfully
    ///
    SUCCESS = 0,
    ///
    /// Gradient check error
    ///
    GRADIENT_ERROR = 1,
    ///
    /// Unknown error
    ///
    UNKNOWN = 2,
}
