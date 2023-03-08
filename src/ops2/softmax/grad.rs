#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::*;

use custos::{Device, Shape, Buffer};

pub trait SoftmaxGrad<T, S: Shape = ()>: Device {
    fn softmax_grad(
        &self,
        samples: usize,
        features: usize,
        x_grad: &mut Buffer<T, Self, S>,
        out: &Buffer<T, Self, S>,
        out_grad: &Buffer<T, Self, S>,
    );
}
