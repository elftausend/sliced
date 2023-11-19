mod grad;
pub use grad::*;

#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::*;

use custos::{Buffer, Device, Shape};

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
    /// let device = CPU::<custos::Base>::new();
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
    #[track_caller]
    fn onehot(&self, classes: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}
