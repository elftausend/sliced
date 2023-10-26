mod grad;
use custos::{Buffer, Device, Shape};
pub use grad::*;

#[cfg(feature = "cpu")]
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

/// Calculates the diagflat of an 1D [`Buffer`] without gradients.
pub trait Diagflat<T, IS: Shape = (), OS: Shape = ()>: Device {
    /// Takes the values of [`Buffer`] `x` and puts them diagonally on the `Buffer` `out`.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use sliced::{Diagflat, Buffer, CPU};
    ///
    /// let device = CPU::<custos::Base>::new();
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
