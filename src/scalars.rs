use crate::{Differentiate, Evaluate, Gradient};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct F64(f64);
impl F64 {
    pub const fn new(v: f64) -> Self {
        Self(v)
    }

    pub const fn get(&self) -> f64 {
        self.0
    }
}
impl Evaluate for F64 {
    type Value = f64;

    fn evaluate(&self) -> Self::Value {
        self.get()
    }
}
impl Differentiate for F64 {
    type Derivative = F64;

    fn differentiate(self) -> Self::Derivative {
        F64::new(0.0)
    }
}
impl Gradient for F64 {
    fn gradient(&self) -> Self::Value {
        self.differentiate().evaluate()
    }
}
