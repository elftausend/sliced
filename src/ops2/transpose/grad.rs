#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::*;

use custos::{Buffer, Device, Shape};

pub trait TranposeGrad<T, IS = (), OS = (), D = Self>: Device
where
    IS: Shape,
    OS: Shape,
    D: Device,
{
    fn transpose_grad(
        &self,
        rows: usize,
        cols: usize,
        x_grad: &mut Buffer<T, D, IS>,
        out_grad: &Buffer<T, D, OS>,
    );
}
