#[cfg(feature = "stack")]
use custos::Stack;
use custos::{impl_stack, Buffer, GenericBlas, Shape, CPU};

use super::GemmGrad;

#[cfg(feature = "blas")]
#[cfg(not(feature = "matrixmultiply"))]
#[impl_stack]
impl<T, D, LS, RS, OS> GemmGrad<T, LS, RS, OS, D> for CPU
where
    T: GenericBlas + Default + Copy,
    D: MainMemory,
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
