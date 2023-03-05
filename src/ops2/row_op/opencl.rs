use std::ops::{Add, AddAssign};

use custos::{
    exec_on_cpu::cpu_exec_binary_may_unified, prelude::cpu_exec_binary_may_unified_mut, Buffer,
    OpenCL,
};

use crate::RowOp;

impl<T> RowOp<T> for OpenCL
where
    T: Copy + Default + Add<Output = T> + AddAssign,
{
    #[inline]
    fn row_op<F: Fn(&mut T, T, T) + Copy>(
        &self,
        rows: usize,
        cols: usize,
        lhs: &Buffer<T, Self>,
        rhs: &Buffer<T, Self>,
        f: F,
    ) -> Buffer<T, Self> {
        cpu_exec_binary_may_unified(self, lhs, rhs, |cpu, lhs, rhs| {
            cpu.row_op(rows, cols, lhs, rhs, f)
        })
        .unwrap()
    }

    #[inline]
    fn add_row_mut(
        &self,
        rows: usize,
        cols: usize,
        lhs: &mut Buffer<T, Self>,
        rhs: &Buffer<T, Self>,
    ) {
        cpu_exec_binary_may_unified_mut(self, lhs, rhs, |cpu, lhs, rhs| {
            cpu.add_row_mut(rows, cols, lhs, rhs)
        })
        .unwrap()
    }
}
