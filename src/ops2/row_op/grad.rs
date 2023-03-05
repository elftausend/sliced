#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
use custos::{Buffer, Device, Shape};
#[cfg(feature = "opencl")]
pub use opencl::*;

pub trait RowOpGrad<T, LS: Shape = (), RS: Shape = (), D: Device = Self>: Device {
    fn add_row_grad(
        &self,
        rows: usize,
        cols: usize,
        lhs_grad: &mut Buffer<T, D, LS>,
        rhs_grad: &mut Buffer<T, D, RS>,
        out_grad: &Buffer<T, D, LS>,
    );
    fn add_row_mut_grad(
        &self,
        rows: usize,
        cols: usize,
        rhs_grad: &mut Buffer<T, D, RS>,
        out_grad: &Buffer<T, D, LS>,
    );
}
