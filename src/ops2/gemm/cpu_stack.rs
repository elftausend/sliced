use std::ops::Deref;

use custos::{
    impl_stack, AddOperation, AsNoId, Buffer, Device, GenericBlas, OnDropBuffer, Retrieve,
    Retriever, Shape, CPU,
};

use super::Gemm;

#[cfg(feature = "stack")]
use custos::Stack;

#[cfg(feature = "blas")]
#[cfg(not(feature = "matrixmultiply"))]
#[impl_stack]
impl<T, D, LS, RS, OS, Mods> Gemm<T, LS, RS, OS, D> for CPU<Mods>
where
    T: GenericBlas + Default + Copy + 'static,
    D: Device + 'static,
    D::Base<T, LS>: Deref<Target = [T]>,
    D::Base<T, RS>: Deref<Target = [T]>,
    LS: Shape,
    RS: Shape,
    OS: Shape,
    Mods: Retrieve<Self, T, OS> + AddOperation + 'static,
{
    fn gemm(
        &self,
        m: usize,
        k: usize,
        n: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve(m * n, (lhs, rhs)).unwrap();
        self.add_op(
            (m.no_id(), k.no_id(), n.no_id(), lhs, rhs, &mut out),
            |(m, k, n, lhs, rhs, out)| {
                T::gemm(**m, **n, **k, lhs, rhs, out);
                Ok(())
            },
        )
        .unwrap();
        out
    }
}

#[cfg(feature = "matrixmultiply")]
#[cfg(not(feature = "blas"))]
#[impl_stack]
impl<T, D, LS, RS, OS> Gemm<T, LS, RS, OS, D> for CPU
where
    T: crate::matrix_multiply::MatrixMultiply + Default + Copy,
    D: MainMemory,
    LS: Shape,
    RS: Shape,
    OS: Shape,
{
    #[inline]
    fn gemm(&self, lhs: &Matrix<T, D, LS>, rhs: &Matrix<T, D, RS>) -> Matrix<T, Self, OS> {
        let (m, k) = lhs.dims();
        let n = rhs.cols();

        debug_assert!(k == rhs.rows());

        let mut out = self.retrieve(m * n, (lhs.node.idx, rhs.node.idx));
        T::gemm(m, k, n, lhs, k, 1, rhs, n, 1, &mut out, n, 1);
        (out, m, n).into()
    }
}

#[cfg(not(feature = "matrixmultiply"))]
#[cfg(not(feature = "blas"))]
//#[impl_stack]
impl<T, D, LS, RS, OS> Gemm<T, LS, RS, OS, D> for CPU
where
    T: Default + Copy + core::ops::Mul<Output = T> + core::ops::AddAssign,
    D: Device,
    D::Base<T, LS>: Deref<Target = [T]>,
    D::Base<T, RS>: Deref<Target = [T]>,
    LS: Shape,
    RS: Shape,
    OS: Shape,
{
    #[inline]
    fn gemm(
        &self,
        m: usize,
        k: usize,
        n: usize,
        lhs: &custos::Buffer<T, D, LS>,
        rhs: &custos::Buffer<T, D, RS>,
    ) -> custos::Buffer<T, Self, OS> {
        // compile_error!("Activate blas feature");

        unimplemented!("This gemm isn't available. Please consider activating the 'blas' feature.")
        /*let mut out = self.retrieve(m * n, (lhs, rhs));
        crate::raw_ops::naive_gemm(m, k, n, lhs, rhs, &mut out);
        (out, m, n).into()
        */
    }
}
