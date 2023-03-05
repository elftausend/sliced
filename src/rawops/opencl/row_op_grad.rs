use std::ops::{AddAssign, Mul};

use crate::RowOpGrad;
use custos::{
    opencl::CLBuffer,
    prelude::{cpu_exec_binary_may_unified_mut, One},
    OpenCL,
};

impl<T> RowOpGrad<T> for OpenCL
where
    T: Copy + Default + AddAssign + One + Mul<Output = T>,
{
    fn row_op_grad(
        &self,
        cols: usize,
        lhs: &CLBuffer<T>,
        rhs: &CLBuffer<T>,
        lhs_grad: &mut CLBuffer<T>,
        rhs_grad: &mut CLBuffer<T>,
        out_grad: &CLBuffer<T>,
        lhs_grad_fn: impl Fn(T) -> T,
        rhs_grad_fn: impl Fn(T) -> T,
    ) {
        let cpu = custos::CPU::new();
        use custos::{Buffer, WriteBuf, CPU};

        #[rustfmt::skip]
        custos::cl_cpu_exec_unified_mut!(
            self, cpu,
            lhs, rhs, out_grad
            WRITE_TO<
                lhs_grad, lhs_grad_cpu,
                rhs_grad, rhs_grad_cpu
            >
            cpu.row_op_grad(cols, &lhs, &rhs,
                &mut lhs_grad_cpu, &mut rhs_grad_cpu,
                &out_grad, lhs_grad_fn, rhs_grad_fn
            )
        );
    }

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
