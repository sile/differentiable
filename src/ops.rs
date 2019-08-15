use crate::scalars::F64;
use crate::{Differentiate, Evaluate, Gradient};
use std::ops::{Add as StdAdd, Mul as StdMul};

pub trait EvaluateExt: Evaluate + Sized {
    // fn add<T>(self, rhs: T) -> Add<Self, T> {
    //     Add::new(self, rhs)
    // }

    // fn mul<T>(self, rhs: T) -> Mul<Self, T> {
    //     Mul::new(self, rhs)
    // }

    fn powi(self, n: i32) -> Powi<Self> {
        Powi::new(self, n)
    }
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
impl<L, R> Differentiate for Add<L, R>
where
    L: Differentiate,
    R: Differentiate,
    L::Value: StdAdd<R::Value>,
{
    type Derivative = Add<L::Derivative, R::Derivative>;

    fn differentiate(self) -> Self::Derivative {
        Add::new(self.lhs.differentiate(), self.rhs.differentiate())
    }
}
impl<L, R, T> StdAdd<T> for Add<L, R> {
    type Output = Add<Self, T>;

    fn add(self, rhs: T) -> Self::Output {
        Add::new(self, rhs)
    }
}
impl<L, R, T> StdMul<T> for Add<L, R> {
    type Output = Mul<Self, T>;

    fn mul(self, rhs: T) -> Self::Output {
        Mul::new(self, rhs)
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
impl<L, R> Gradient for Mul<L, R>
where
    L: Gradient,
    R: Gradient,
    L::Value: StdMul<R::Value>,
    MulValue<L, R>: StdAdd<MulValue<L, R>, Output = MulValue<L, R>>,
{
    fn gradient(&self) -> Self::Value {
        let lhs = self.lhs.gradient() * self.rhs.evaluate();
        let rhs = self.lhs.evaluate() * self.rhs.gradient();
        lhs + rhs
    }
}
impl<L, R> Differentiate for Mul<L, R>
where
    L: Differentiate + Clone,
    R: Differentiate + Clone,
    L::Value: StdMul<R::Value>,
    MulValue<L, R>: StdAdd<MulValue<L, R>, Output = MulValue<L, R>>,
{
    type Derivative = Add<Mul<L::Derivative, R>, Mul<L, R::Derivative>>;

    fn differentiate(self) -> Self::Derivative {
        let lhs = Mul::new(self.lhs.clone().differentiate(), self.rhs.clone());
        let rhs = Mul::new(self.lhs, self.rhs.differentiate());
        Add::new(lhs, rhs)
    }
}
impl<L, R, T> StdAdd<T> for Mul<L, R> {
    type Output = Add<Self, T>;

    fn add(self, rhs: T) -> Self::Output {
        Add::new(self, rhs)
    }
}
impl<L, R, T> StdMul<T> for Mul<L, R> {
    type Output = Mul<Self, T>;

    fn mul(self, rhs: T) -> Self::Output {
        Mul::new(self, rhs)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Powi<T> {
    operand: T,
    n: i32,
}
impl<T> Powi<T> {
    pub const fn new(operand: T, n: i32) -> Self {
        Self { operand, n }
    }
}
impl<T> Evaluate for Powi<T>
where
    T: Evaluate<Value = f64>,
{
    type Value = f64;

    fn evaluate(&self) -> Self::Value {
        self.operand.evaluate().powi(self.n)
    }
}
impl<T> Gradient for Powi<T>
where
    T: Evaluate<Value = f64>,
{
    fn gradient(&self) -> Self::Value {
        f64::from(self.n) * self.operand.evaluate().powi(self.n - 1)
    }
}
impl<T> Differentiate for Powi<T>
where
    T: Evaluate<Value = f64>,
{
    type Derivative = Mul<F64, Powi<T>>;

    fn differentiate(self) -> Self::Derivative {
        F64::new(self.n as f64).mul(self.operand.powi(self.n - 1))
    }
}
impl<T, U> StdAdd<U> for Powi<T> {
    type Output = Add<Self, U>;

    fn add(self, rhs: U) -> Self::Output {
        Add::new(self, rhs)
    }
}
impl<T, U> StdMul<U> for Powi<T> {
    type Output = Mul<Self, U>;

    fn mul(self, rhs: U) -> Self::Output {
        Mul::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalars::F64;

    #[test]
    fn add_works() {
        let a = F64::new(2.0);
        let b = F64::new(5.0);
        let c = a + b;
        assert_eq!(c.evaluate(), 7.0);
        assert_eq!(c.gradient(), 0.0);
        assert_eq!(c.differentiate().evaluate(), 0.0);
    }

    #[test]
    fn mul_works() {
        let a = F64::new(2.0);
        let b = F64::new(5.0);
        let c = F64::new(3.0);
        let d = (a + b) * c;
        assert_eq!(d.evaluate(), 21.0);
        assert_eq!(d.gradient(), 0.0);
        assert_eq!(d.differentiate().evaluate(), 0.0);
    }

    #[test]
    fn powi_works() {
        let v = F64::new(2.0).powi(4);
        assert_eq!(v.evaluate(), 16.0);
        assert_eq!(v.gradient(), 32.0);
        assert_eq!(v.differentiate().evaluate(), 32.0);

        let a = F64::new(3.0);
        let b = F64::new(1.0);
        let c = F64::new(2.0);
        let d = a * (b + c).powi(3);
        assert_eq!(d.evaluate(), 81.0);
        assert_eq!(d.gradient(), 81.0);
        assert_eq!(d.differentiate().evaluate(), 81.0);
    }
}
