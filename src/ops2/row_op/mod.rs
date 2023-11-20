mod grad;
use std::ops::Add;

use custos::{Buffer, Device, Shape, Resolve, Eval, MayToCLSource, Combiner};
pub use grad::*;

#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::*;

pub trait RowOp<T, LS: Shape = (), RS: Shape = (), D: Device = Self>: Device {
    #[track_caller]
    fn row_op<O: Eval<T> + MayToCLSource>(
        &self,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
        f: impl Fn(Resolve<T>, Resolve<T>) -> O,
    ) -> Buffer<T, Self, LS>;

    #[inline]
    #[track_caller]
    fn add_row(
        &self,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, Self, LS>
    where
        T: Add<Output = T>,
    {
        self.row_op(cols, lhs, rhs, |a, b| a.add(b))
    }
    #[track_caller]
    fn add_row_mut(
        &self,
        rows: usize,
        cols: usize,
        lhs: &mut Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    );
}
