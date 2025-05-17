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

use custos::{Buffer, Device, Shape, Unit};

pub trait Softmax<T: Unit, S: Shape = (), D: Device = Self>: Device {
    fn softmax(&self, samples: usize, features: usize, x: &Buffer<T, D, S>) -> Buffer<T, Self, S>;
}
