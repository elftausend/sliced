use std::ops::AddAssign;

use custos::{prelude::CLBuffer, CDatatype, OpenCL, Buffer};

use crate::{assign_or_set::Assign, cl_gemm, GemmGrad, Transpose};

impl<T: CDatatype + AddAssign> GemmGrad<T> for OpenCL {
    fn gemm_grad(
        &self,
        m: usize,
        k: usize,
        n: usize,
        lhs: &Buffer<T, Self>,
        rhs: &Buffer<T, Self>,
        lhs_grad: &mut Buffer<T, Self>,
        rhs_grad: &mut Buffer<T, Self>,
        out_grad: &Buffer<T, Self>,
    ) {
        cl_gemm::<T, Assign>(
            self,
            m,
            n,
            k,
            out_grad,
            &self.transpose(k, n, rhs),
            lhs_grad,
        )
        .unwrap();
        cl_gemm::<T, Assign>(
            self,
            k,
            m,
            n,
            &self.transpose(m, k, lhs),
            out_grad,
            rhs_grad,
        )
        .unwrap();
    }
}
