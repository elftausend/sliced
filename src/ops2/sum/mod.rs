mod grad;
use custos::{Buffer, Device, Shape};
pub use grad::*;

#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::*;

pub trait Sum<T, S: Shape = (), D: Device = Self>: Device {
    fn sum(&self, x: &Buffer<T, D, S>) -> T;
}

pub trait SumRows<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn sum_rows(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub trait SumCols<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn sum_cols(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}
