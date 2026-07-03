#[cfg(any(feature = "cpu", feature = "stack"))]
mod cpu;
#[cfg(any(feature = "cpu", feature = "stack"))]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::*;

use custos::{Buffer, Device, Shape, Unit};

pub trait DiagflatGrad<T: Unit, IS: Shape = (), OS: Shape = ()>: Device {
    fn diagflat_grad(&self, x_grad: &mut Buffer<T, Self, IS>, out_grad: &Buffer<T, Self, OS>);
}
