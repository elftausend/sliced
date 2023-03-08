#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::*;

use custos::{Shape, Buffer, Device};

pub trait DiagflatGrad<T, IS: Shape = (), OS: Shape = ()>: Device {
    fn diagflat_grad(&self, x_grad: &mut Buffer<T, Self, IS>, out_grad: &Buffer<T, Self, OS>);
}