//pub mod combinators;
pub mod ops;
pub mod scalars;

pub trait Evaluate {
    type Value;

    fn evaluate(&self) -> Self::Value;
}

pub trait Differentiate: Evaluate {
    type Derivative: Evaluate<Value = Self::Value>;

    fn differentiate(self) -> Self::Derivative;
}

pub trait Gradient: Differentiate {
    fn gradient(&self) -> Self::Value;
}
