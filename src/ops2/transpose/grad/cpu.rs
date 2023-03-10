use std::ops::AddAssign;

use custos::{Buffer, MainMemory, Shape, CPU};

use crate::{assign_or_set::Assign, slice_transpose, TranposeGrad};

impl<T, IS: Shape, OS: Shape, D: MainMemory> TranposeGrad<T, IS, OS, D> for CPU
where
    T: Default + Copy + AddAssign,
    IS: Shape,
    OS: Shape,
    D: MainMemory,
{
    #[inline]
    fn transpose_grad(
        &self,
        rows: usize,
        cols: usize,
        x_grad: &mut Buffer<T, D, IS>,
        out_grad: &Buffer<T, D, OS>,
    ) {
        slice_transpose::<T, Assign>(rows, cols, out_grad, x_grad)
    }
}
