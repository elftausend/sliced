use std::ops::AddAssign;

use custos::{prelude::CLBuffer, CDatatype, OpenCL};

use crate::{assign_or_set::Assign, cl_gemm, GemmGrad, Transpose};

impl<T: CDatatype + AddAssign> GemmGrad<T> for OpenCL {
    fn gemm_grad(
        &self,
        m: usize,
        k: usize,
        n: usize,
        lhs: &CLBuffer<T>,
        rhs: &CLBuffer<T>,
        lhs_grad: &mut CLBuffer<T>,
        rhs_grad: &mut CLBuffer<T>,
        out_grad: &CLBuffer<T>,
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
