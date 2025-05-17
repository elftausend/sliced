#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::*;

use custos::{Buffer, Device, Shape, Unit};

pub trait MaxRowsGrad<T: Unit, IS: Shape = (), OS: Shape = ()>: Device {
    fn max_rows_grad(
        &self,
        cols: usize,
        out: &Buffer<T, Self, OS>,
        x: &Buffer<T, Self, IS>,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    );
}

pub trait MaxColsGrad<T: Unit, IS: Shape = (), OS: Shape = ()>: Device {
    fn max_cols_grad(
        &self,
        cols: usize,
        out: &Buffer<T, Self, OS>,
        x: &Buffer<T, Self, IS>,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    );
}
