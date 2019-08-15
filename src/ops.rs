use crate::scalars::F64;
use crate::{Differentiate, Evaluate, Gradient};
use std::ops::{Add as StdAdd, Mul as StdMul};

pub trait EvaluateExt: Evaluate + Sized {
    fn add<T>(self, rhs: T) -> Add<Self, T> {
        Add::new(self, rhs)
    }

    fn mul<T>(self, rhs: T) -> Mul<Self, T> {
        Mul::new(self, rhs)
    }

    // fn powi(self, n: i32) -> Powi<Self> {
    //     Powi::new(self, n)
    // }
}
impl<T: Evaluate + Sized> EvaluateExt for T {}

#[derive(Debug, Clone, Copy)]
pub struct Add<L, R> {
    lhs: L,
    rhs: R,
}
impl<L, R> Add<L, R> {
    pub const fn new(lhs: L, rhs: R) -> Self {
        Self { lhs, rhs }
    }
}
impl<L, R> Evaluate for Add<L, R>
where
    L: Evaluate,
    R: Evaluate,
    L::Value: StdAdd<R::Value>,
{
    type Value = <L::Value as StdAdd<R::Value>>::Output;

    fn evaluate(&self) -> Self::Value {
        self.lhs.evaluate() + self.rhs.evaluate()
    }
}
impl<L, R> Differentiate for Add<L, R>
where
    L: Differentiate,
    R: Differentiate,
    L::Value: StdAdd<R::Value>,
{
    type Derivative = Add<L::Derivative, R::Derivative>;

    fn differentiate(self) -> Self::Derivative {
        self.lhs.differentiate().add(self.rhs.differentiate())
    }
}
impl<L, R> Gradient for Add<L, R>
where
    L: Gradient,
    R: Gradient,
    L::Value: StdAdd<R::Value>,
{
    fn gradient(&self) -> Self::Value {
        self.lhs.gradient() + self.rhs.gradient()
    }
}

pub type MulValue<L, R> = <Mul<L, R> as Evaluate>::Value;

#[derive(Debug, Clone, Copy)]
pub struct Mul<L, R> {
    lhs: L,
    rhs: R,
}
impl<L, R> Mul<L, R> {
    pub const fn new(lhs: L, rhs: R) -> Self {
        Self { lhs, rhs }
    }
}
impl<L, R> Evaluate for Mul<L, R>
where
    L: Evaluate,
    R: Evaluate,
    L::Value: StdMul<R::Value>,
{
    type Value = <L::Value as StdMul<R::Value>>::Output;

    fn evaluate(&self) -> Self::Value {
        self.lhs.evaluate() * self.rhs.evaluate()
    }
}
impl<L, R> Differentiate for Mul<L, R>
where
    L: Differentiate + Clone,
    R: Differentiate + Clone,
    L::Value: StdMul<R::Value>,
    MulValue<L::Derivative, R>: StdAdd<MulValue<L, R::Derivative>, Output = MulValue<L, R>>,
{
    type Derivative = Add<Mul<L::Derivative, R>, Mul<L, R::Derivative>>;

    fn differentiate(self) -> Self::Derivative {
        let lhs = self.lhs.clone().differentiate().mul(self.rhs.clone());
        let rhs = self.lhs.mul(self.rhs.differentiate());
        lhs.add(rhs)
    }
}
impl<L, R> Gradient for Mul<L, R>
where
    L: Gradient + Clone,
    R: Gradient + Clone,
    L::Value: StdMul<R::Value>,
    MulValue<L::Derivative, R>: StdAdd<MulValue<L, R::Derivative>, Output = MulValue<L, R>>,
{
    fn gradient(&self) -> Self::Value {
        let lhs = self.lhs.gradient() * self.rhs.evaluate();
        let rhs = self.lhs.evaluate() * self.rhs.gradient();
        lhs + rhs
    }
}

// #[derive(Debug, Clone, Copy)]
// pub struct Powi<T> {
//     operand: T,
//     n: i32,
// }
// impl<T> Powi<T> {
//     pub const fn new(operand: T, n: i32) -> Self {
//         Self { operand, n }
//     }
// }
// impl<T> Expr for Powi<T>
// where
//     T: Expr<Value = f64>,
// {
//     type Value = f64;

//     fn evaluate(&self) -> f64 {
//         self.operand.evaluate().powi(self.n)
//     }
// }
// impl<T> Grad for Powi<T>
// where
//     T: Expr<Value = f64>,
// {
//     type Gradient = Mul<F64, Powi<T>>;

//     fn grad(self) -> Self::Gradient {
//         F64::new(self.n as f64).mul(self.operand.powi(self.n - 1))
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalars::F64;

    #[test]
    fn add_works() {
        let a = F64::new(2.0);
        let b = F64::new(5.0);
        let c = a.add(b);
        assert_eq!(c.evaluate(), 7.0);
        assert_eq!(c.gradient(), 0.0);
    }

    // #[test]
    // fn powi_works() {
    //     let v = F64::new(2.0).powi(4);
    //     assert_eq!(v.evaluate(), 16.0);
    // }
}
