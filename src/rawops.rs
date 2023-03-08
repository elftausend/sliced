use std::{
    fmt::Display,
    ops::{Div, Sub},
};

use custos::{Buffer, Combiner, Device, Eval, MayDim2, Resolve, Shape};

mod cpu;

#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "stack")]
mod stack;
#[cfg(feature = "stack")]
pub use stack::*;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(feature = "opencl")]
pub use opencl::*;

pub trait BinaryEWMayGrad<T, S: Shape = (), D: Device = Self>: Device {
    fn binary_ew_w_grad<FO, LO, RO>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        forward_fn: impl Fn(Resolve<T>, Resolve<T>) -> FO,
        lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LO,
        rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RO,
    ) -> Buffer<T, Self, S>
    where
        FO: Eval<T> + ToString,
        LO: Eval<T> + ToString,
        RO: Eval<T> + ToString;
}

pub trait RandOp<T, S: Shape = (), D: Device = Self>: Device {
    fn rand(&self, x: &mut Buffer<T, D, S>, lo: T, hi: T);
}

pub trait Softmax<T, S: Shape = (), D: Device = Self>: Device {
    fn softmax(&self, samples: usize, features: usize, x: &Buffer<T, D, S>) -> Buffer<T, Self, S>;
}

pub trait SoftmaxGrad<T, S: Shape = ()>: Device {
    fn softmax_grad(
        &self,
        samples: usize,
        features: usize,
        x_grad: &mut Buffer<T, Self, S>,
        out: &Buffer<T, Self, S>,
        out_grad: &Buffer<T, Self, S>,
    );
}

//pub trait SumOp

/// Provides the onehot encoding operation.
pub trait Onehot<T, IS: Shape = (), OS: Shape = ()>: Device {
    /// Onehot encodes a `Buffer` of classes.
    ///
    /// # Example
    ///
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    ///
    /// use sliced::{CPU, Onehot, Buffer};
    ///
    /// let device = CPU::new();
    ///
    /// let classes = Buffer::from((&device, [1, 0, 3, 2]));
    /// let onehot = device.onehot(&classes);
    ///
    /// assert_eq!([
    ///     0, 1, 0, 0,
    ///     1, 0, 0, 0,
    ///     0, 0, 0, 1,
    ///     0, 0, 1, 0
    /// ], &*onehot);
    /// ```
    fn onehot(&self, classes: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

/// Calculates the diagflat of an 1D [`Buffer`] without gradients.
pub trait Diagflat<T, IS: Shape = (), OS: Shape = ()>: Device {
    /// Takes the values of [`Buffer`] `x` and puts them diagonally on the `Buffer` `out`.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use sliced::{Diagflat, Buffer, CPU};
    ///
    /// let device = CPU::new();
    /// let x = Buffer::from((&device, [1, 2, 7, -1, -2]));
    /// let out: Buffer<i32> = device.diagflat(&x);
    ///
    /// assert_eq!(&*out, [
    ///     1, 0, 0, 0, 0,
    ///     0, 2, 0, 0, 0,
    ///     0, 0, 7, 0, 0,
    ///     0, 0, 0, -1, 0,
    ///     0, 0, 0, 0, -2,
    /// ]);
    ///
    /// ```
    fn diagflat(&self, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

pub trait DiagflatGrad<T, IS: Shape = (), OS: Shape = ()>: Device {
    fn diagflat_grad(&self, x_grad: &mut Buffer<T, Self, IS>, out_grad: &Buffer<T, Self, OS>);
}
