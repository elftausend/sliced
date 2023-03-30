#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
use custos::{Buffer, Device, Shape, Eval, Resolve};
#[cfg(feature = "opencl")]
pub use opencl::*;

pub trait ColOpGrad<T, LS: Shape = (), RS: Shape = (), D: Device = Self>: Device {
    fn row_op_grad<LhsGrad, RhsGrad>(
        &self,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
        lhs_grad: &mut Buffer<T, D, LS>,
        rhs_grad: &mut Buffer<T, D, RS>,
        out_grad: &Buffer<T, D, LS>,
        lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LhsGrad,
        rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RhsGrad,
    ) 
    where
        LhsGrad: Eval<T> + ToString,
        RhsGrad: Eval<T> + ToString;
}
