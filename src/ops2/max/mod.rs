mod grad;
pub use grad::*;

#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
use custos::{Buffer, Device, Shape};
#[cfg(feature = "opencl")]
pub use opencl::*;

pub trait Max<T, S: Shape = (), D: Device = Self>: Device {
    fn max(&self, x: &Buffer<T, D, S>) -> T;
}

pub trait MaxRows<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn max_rows(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub trait MaxCols<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn max_cols(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}
