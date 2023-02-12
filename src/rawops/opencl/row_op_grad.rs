use std::ops::AddAssign;

use crate::RowOpGrad;
use custos::{opencl::CLBuffer, prelude::cpu_exec_binary_may_unified_mut, OpenCL};

impl<T: Copy + Default + AddAssign> RowOpGrad<T> for OpenCL {
    fn add_row_grad(
        &self,
        rows: usize,
        cols: usize,
        lhs_grad: &mut CLBuffer<T>,
        rhs_grad: &mut CLBuffer<T>,
        out_grad: &CLBuffer<T>,
    ) {
        todo!()
    }

    #[inline]
    fn add_row_mut_grad(
        &self,
        rows: usize,
        cols: usize,
        rhs_grad: &mut CLBuffer<T>,
        out_grad: &CLBuffer<T>,
    ) {
        cpu_exec_binary_may_unified_mut(self, rhs_grad, out_grad, |cpu, rhs_grad, out_grad| {
            cpu.add_row_mut_grad(rows, cols, rhs_grad, out_grad)
        })
        .unwrap();
    }
}
