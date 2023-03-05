mod grad;
pub use grad::*;

#[cfg(any(feature = "cpu", feature = "stack"))]
mod cpu_stack;

#[cfg(any(feature = "cpu", feature = "stack"))]
pub use cpu_stack::*;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(feature = "opencl")]
pub use opencl::*;

use core::fmt::Display;

use custos::{Buffer, Combiner, Device, Eval, Resolve, Shape};

pub trait BinaryElementWise<T, S: Shape = (), D: Device = Self>: Device {
    fn binary_ew<O>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        f: impl Fn(Resolve<T>, Resolve<T>) -> O,
    ) -> Buffer<T, Self, S>
    where
        O: Eval<T> + ToString;

    #[inline]
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>
    where
        T: Display + Eval<T> + core::ops::Add<T, Output = T>,
    {
        self.binary_ew(lhs, rhs, |lhs, rhs| lhs.add(rhs))
    }

    #[inline]
    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>
    where
        T: Display + Eval<T> + core::ops::Mul<T, Output = T>,
    {
        self.binary_ew(lhs, rhs, |lhs, rhs| lhs.mul(rhs))
    }

    #[inline]
    fn div(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>
    where
        T: Display + Eval<T> + core::ops::Div<T, Output = T>,
    {
        self.binary_ew(lhs, rhs, |lhs, rhs| lhs.div(rhs))
    }

    #[inline]
    fn sub(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>
    where
        T: Display + Eval<T> + core::ops::Sub<T, Output = T>,
    {
        self.binary_ew(lhs, rhs, |lhs, rhs| lhs.sub(rhs))
    }
}
