mod grad;
use custos::{Buffer, Device, MayDim2, Shape};
pub use grad::*;

#[cfg(feature = "cpu")]
mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::*;

use crate::assign_or_set::{AssignOrSet, Set};

pub trait Transpose<T, IS: Shape = (), OS: Shape = (), D: Device = Self, AOS: AssignOrSet<T> = Set>:
    Device
{
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub trait Transpose2<
    T,
    const ROWS: usize = 0,
    const COLS: usize = 0,
    IS: MayDim2<ROWS, COLS> = (),
    OS: MayDim2<COLS, ROWS> = (),
    D: Device = Self,
>: Device
{
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}
