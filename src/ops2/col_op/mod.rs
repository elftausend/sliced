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

use core::ops::{Div, Sub};

use custos::{Buffer, Device, Shape};

pub trait ColOp<T, LS: Shape = (), RS: Shape = (), D: Device = Self>: Device {
    fn col_op<F>(
        &self,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
        f: F,
    ) -> Buffer<T, Self, LS>
    where
        F: Fn(T, T) -> T + Copy;

    #[inline]
    fn sub_cols(
        &self,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, Self, LS>
    where
        T: Sub<Output = T>,
    {
        self.col_op(cols, lhs, rhs, |lhs, rhs| lhs - rhs)
    }

    #[inline]
    fn div_cols(
        &self,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, Self, LS>
    where
        T: Div<Output = T>,
    {
        self.col_op(cols, lhs, rhs, |lhs, rhs| lhs / rhs)
    }
}
