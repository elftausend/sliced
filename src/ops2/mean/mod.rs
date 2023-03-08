mod grad;
use custos::{Device, Shape, Buffer};
pub use grad::*;

#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::*;

pub trait Mean<T, S: Shape>: Device {
    fn mean(&self, x: &Buffer<T, Self, S>) -> T;
}

/// Calculates the mean of every column (while interacting with the rows).
pub trait MeanRows<T, IS: Shape = (), OS: Shape = ()>: Device {
    /// Calculates the mean of every column (while interacting with the rows).
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use sliced::{CPU, MeanRows, Buffer};
    ///
    /// let device = CPU::new();
    ///
    /// let to_mean_rows = Buffer::from((&device, [
    ///     2, 1, 3,
    ///     1, 2, 3,
    ///     3, 1, 1,
    ///     2, 4, 1,
    /// ]));
    /// let mean_rows: Buffer<_> = device.mean_rows(3, &to_mean_rows);
    /// assert_eq!(&*mean_rows, [2, 2, 2]);
    ///
    /// ```
    fn mean_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

/// Calculates the mean of every row (while interacting with the columns).
pub trait MeanCols<T, IS: Shape = (), OS: Shape = ()>: Device {
    /// Calculates the mean of every row (while interacting with the columns).
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use sliced::{Buffer, CPU, MeanCols};
    ///
    /// let device = CPU::new();
    ///
    /// let to_mean_cols = Buffer::from((&device, [
    ///     1, 4, 1, 2,
    ///     2, 2, 2, 2,
    ///     8, 1, 2, 1,
    /// ]));
    ///
    /// let mean_cols: Buffer<_>  = device.mean_cols(4, &to_mean_cols);
    /// assert_eq!(&*mean_cols, [2, 2, 3])
    /// ```
    fn mean_cols(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}