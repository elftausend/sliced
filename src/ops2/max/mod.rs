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

pub trait Max<T, S: Shape = (), D: Device = Self>: Device {
    #[track_caller]
    fn max(&self, x: &Buffer<T, D, S>) -> T;
}

pub trait MaxRows<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    #[track_caller]
    fn max_rows(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub trait MaxCols<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    #[track_caller]
    fn max_cols(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}
