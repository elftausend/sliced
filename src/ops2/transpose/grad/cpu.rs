use std::ops::{AddAssign, Deref, DerefMut};

use custos::{Buffer, Device, Shape, CPU, OnDropBuffer};

use crate::{assign_or_set::Assign, slice_transpose, TranposeGrad};

impl<T, IS, OS, D, Mods: OnDropBuffer> TranposeGrad<T, IS, OS, D> for CPU<Mods>
where
    T: Default + Copy + AddAssign,
    IS: Shape,
    OS: Shape,
    D: Device,
    D::Data<T, IS>: Deref<Target = [T]> + DerefMut,
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
