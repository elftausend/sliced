use core::ops::{Deref, DerefMut};

#[cfg(feature = "stack")]
use custos::Stack;
#[cfg(feature = "cpu")]
use custos::CPU;
use custos::{impl_stack, Buffer, Device, GenericBlas, HasId, Shape, Unit, WrappedData};
#[cfg(not(feature = "matrixmultiply"))]
#[cfg(not(feature = "blas"))]
use custos::{AddOperation, Retrieve};

#[cfg(not(feature = "matrixmultiply"))]
#[cfg(not(feature = "blas"))]
use crate::{assign_or_set::Set, Transposed};

use super::GemmGrad;

#[cfg(feature = "blas")]
#[cfg(not(feature = "matrixmultiply"))]
#[impl_stack]
impl<T, D, LS, RS, OS, Mods: WrappedData> GemmGrad<T, LS, RS, OS, D> for CPU<Mods>
where
    T: Unit + GenericBlas + Default + Copy,
    D: Device,
    D::Base<T, LS>: Deref<Target = [T]> + DerefMut,
    D::Base<T, RS>: Deref<Target = [T]> + DerefMut,
    D::Base<T, OS>: Deref<Target = [T]>,
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
        if lhs.requires_grad() {
            T::gemmT(m, k, n, out_grad, rhs, lhs_grad);
        }
        if rhs.requires_grad() {
            T::Tgemm(k, n, m, lhs, out_grad, rhs_grad);
        }
    }
}

#[cfg(not(feature = "blas"))]
#[cfg(not(feature = "matrixmultiply"))]
#[impl_stack]
impl<T, D, LS, RS, OS, Mods> GemmGrad<T, LS, RS, OS, D> for CPU<Mods>
where
    T: Unit + Default + Copy + core::ops::AddAssign + core::ops::Mul<Output = T>,
    D: Device + 'static,
    D::Base<T, LS>: Deref<Target = [T]> + DerefMut,
    D::Base<T, RS>: Deref<Target = [T]> + DerefMut,
    D::Base<T, OS>: Deref<Target = [T]>,
    LS: Shape,
    RS: Shape,
    OS: Shape,
    Mods: WrappedData
        + AddOperation
        + Retrieve<Self, T, Transposed<RS>>
        + Retrieve<Self, T, Transposed<LS>>
        + 'static,
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
        use crate::Transpose;
        if lhs.requires_grad() {
            let rhs = Transpose::<T, RS, Transposed<RS>, D, Set>::transpose(self, k, n, rhs);
            rawsliced::naive_gemm(m, n, k, out_grad, &rhs, lhs_grad);
        }
        if rhs.requires_grad() {
            let lhs = Transpose::<T, LS, Transposed<LS>, D, Set>::transpose(self, m, k, lhs);
            rawsliced::naive_gemm(k, m, n, &lhs, out_grad, rhs_grad);
        }
    }
}
