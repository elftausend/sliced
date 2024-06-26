use std::ops::{Add, AddAssign};

use custos::{
    exec_on_cpu::cpu_exec_binary_may_unified, prelude::cpu_exec_binary_may_unified_mut, Buffer,
    OnDropBuffer, OpenCL, Retrieve, UnifiedMemChain,
};

use crate::RowOp;

impl<Mods, T> RowOp<T> for OpenCL<Mods>
where
    T: Copy + Default + Add<Output = T> + AddAssign + 'static,
    Mods: Retrieve<Self, T> + UnifiedMemChain<Self> + 'static,
{
    #[inline]
    fn row_op<F: Fn(&mut T, T, T) + Copy>(
        &self,
        cols: usize,
        lhs: &Buffer<T, Self>,
        rhs: &Buffer<T, Self>,
        f: F,
    ) -> Buffer<T, Self> {
        cpu_exec_binary_may_unified(self, lhs, rhs, |cpu, lhs, rhs| {
            cpu.row_op(cols, lhs, rhs, f)
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
