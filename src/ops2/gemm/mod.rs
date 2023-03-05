mod grad;
use custos::{Buffer, Device, Shape};
pub use grad::*;

#[cfg(any(feature = "cpu", feature = "stack"))]
mod cpu_stack;

#[cfg(any(feature = "cpu", feature = "stack"))]
pub use cpu_stack::*;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(feature = "opencl")]
pub use opencl::*;

pub trait Gemm<T, LS: Shape = (), RS: Shape = (), OS: Shape = (), D: Device = Self>:
    Device
{
    fn gemm(
        &self,
        m: usize,
        k: usize,
        n: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, Self, OS>;
}
