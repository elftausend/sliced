#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::*;

use custos::{Device, Shape, Buffer};
pub trait MeanRowsGrad<T, IS: Shape = (), OS: Shape = ()>: Device {
    fn mean_rows_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    );
}

pub trait MeanColsGrad<T, IS: Shape = (), OS: Shape = ()>: Device {
    fn mean_cols_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    );
}
