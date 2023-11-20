use std::ops::{Deref, DerefMut};

#[cfg(feature = "stack")]
use custos::Stack;
use custos::{impl_stack, Buffer, Device, GenericBlas, OnDropBuffer, Shape, CPU};

use super::GemmGrad;

#[cfg(feature = "blas")]
#[cfg(not(feature = "matrixmultiply"))]
#[impl_stack]
impl<T, D, LS, RS, OS, Mods: OnDropBuffer> GemmGrad<T, LS, RS, OS, D> for CPU<Mods>
where
    T: GenericBlas + Default + Copy,
    D: Device,
    D::Data<T, LS>: Deref<Target = [T]> + DerefMut,
    D::Data<T, RS>: Deref<Target = [T]> + DerefMut,
    D::Data<T, OS>: Deref<Target = [T]>,
    LS: Shape,
    RS: Shape,
    OS: Shape,
{
    #[inline]
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
    ) {
        T::gemmT(m, k, n, out_grad, rhs, lhs_grad);
        T::Tgemm(k, n, m, lhs, out_grad, rhs_grad);
    }
}
