use std::ops::{AddAssign, Deref};

use custos::{Buffer, Device, Shape, CPU};

use crate::{assign_or_set::Assign, slice_transpose, TranposeGrad};

impl<T, IS, OS, D> TranposeGrad<T, IS, OS, D> for CPU
where
    T: Default + Copy + AddAssign,
    IS: Shape,
    OS: Shape,
    D: Device,
    D::Data<T, IS>: Deref<Target = [T]>,
    D::Data<T, OS>: Deref<Target = [T]>,
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
