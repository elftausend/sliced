#[cfg(any(feature = "cpu", feature = "stack"))]
mod cpu_stack;

#[cfg(any(feature = "cpu", feature = "stack"))]
pub use cpu_stack::*;

#[cfg(feature = "opencl")]
mod opencl;

use custos::{impl_nnapi_op, Buffer, Device, Shape};
#[cfg(feature = "opencl")]
pub use opencl::*;

#[impl_nnapi_op(None)]
pub trait GemmGrad<T, LS: Shape = (), RS: Shape = (), OS: Shape = (), D: Device = Self>:
    Device
{
    fn gemm_grad(
        &self,
        m: usize,
        k: usize,
        n: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
        lhs_grad: &mut Buffer<T, D, LS>,
        rhs_grad: &mut Buffer<T, D, RS>,
        out_grad: &Buffer<T, D, OS>,
    );
}
