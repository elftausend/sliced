use custos::{prelude::cpu_exec_binary_may_unified, Buffer, OpenCL};

use crate::ColOp;

impl<T: Copy + Default> ColOp<T> for OpenCL {
    #[inline]
    fn col_op<F>(
        &self,
        cols: usize,
        lhs: &Buffer<T, Self>,
        rhs: &Buffer<T, Self>,
        f: F,
    ) -> Buffer<T, Self>
    where
        F: Fn(T, T) -> T + Copy,
    {
        cpu_exec_binary_may_unified(self, lhs, rhs, |cpu, lhs, rhs| {
            cpu.col_op(cols, lhs, rhs, f)
        })
        .unwrap()
    }
}
