//pub mod combinators;
pub mod ops;
pub mod scalars;

pub trait Evaluate {
    type Value;

    fn evaluate(&self) -> Self::Value;
}

pub trait Gradient: Evaluate {
    fn gradient(&self) -> Self::Value;
}

pub trait Differentiate: Evaluate {
    type Derivative: Evaluate<Value = Self::Value>;

    fn differentiate(self) -> Self::Derivative;
}
