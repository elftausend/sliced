#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
use custos::{Buffer, Device, Shape};
#[cfg(feature = "opencl")]
pub use opencl::*;

pub trait SumRowsGrad<T, IS: Shape = (), OS: Shape = ()>: Device {
    fn sum_rows_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    );
}

pub trait SumColsGrad<T, IS: Shape = (), OS: Shape = ()>: Device {
    fn sum_cols_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, IS>,
    );
}
