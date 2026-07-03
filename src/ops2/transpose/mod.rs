mod grad;
use core::marker::PhantomData;

use custos::{Buffer, Device, Dim2, MayDim2, Shape, Unit};
pub use grad::*;

#[cfg(any(feature = "cpu", feature = "stack"))]
mod cpu;
#[cfg(any(feature = "cpu", feature = "stack"))]
pub use cpu::*;

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::*;

use crate::assign_or_set::{AssignOrSet, Set};

pub trait Transpose<
    T: Unit,
    IS: Shape = (),
    OS: Shape = (),
    D: Device = Self,
    AOS: AssignOrSet<T> = Set,
>: Device
{
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub struct Transposed<S> {
    pd: PhantomData<S>,
}

impl<S: Shape> Shape for Transposed<S> {
    type Array<T> = S::Array<T>;

    fn new<T: Copy + Default>() -> Self::Array<T> {
        S::new()
    }

    #[cfg(feature = "std")]
    fn dims() -> Vec<usize> {
        todo!()
    }
}

// pub trait Transpose2<
//     T: Unit,
//     const ROWS: usize = 0,
//     const COLS: usize = 0,
//     IS: MayDim2<ROWS, COLS> = (),
//     OS: MayDim2<COLS, ROWS> = (),
//     D: Device = Self,
// >: Device
// {
//     fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
// }
