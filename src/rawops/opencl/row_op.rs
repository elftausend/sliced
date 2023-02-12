use std::ops::{Add, AddAssign};

use custos::{OpenCL, exec_on_cpu::cpu_exec_binary_may_unified, prelude::cpu_exec_binary_may_unified_mut};

use crate::RowOp;


impl<T: Copy + Default + Add<Output = T> + AddAssign> RowOp<T> for OpenCL {
    #[inline]
    fn add_row(
        &self,
        rows: usize,
        cols: usize,
        lhs: &custos::Buffer<T, Self>,
        rhs: &custos::Buffer<T, Self>,
    ) -> custos::Buffer<T, Self> 
    {
        cpu_exec_binary_may_unified(self, lhs, rhs, |cpu, lhs, rhs| cpu.add_row(rows, cols, lhs, rhs)).unwrap()
    }

    #[inline]
    fn add_row_mut(
        &self,
        rows: usize,
        cols: usize,
        lhs: &mut custos::Buffer<T, Self, ()>,
        rhs: &custos::Buffer<T, Self, ()>,
    ) {
        cpu_exec_binary_may_unified_mut(self, lhs, rhs, |cpu, lhs, rhs| cpu.add_row_mut(rows, cols, lhs, rhs)).unwrap()
    }
}