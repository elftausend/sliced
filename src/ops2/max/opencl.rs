use custos::{
    exec_on_cpu::cpu_exec_reduce_may_unified,
    prelude::{cpu_exec_unary_may_unified, Number},
    Buffer, OnDropBuffer, OpenCL, Retrieve, UnifiedMemChain,
};

use crate::{Max, MaxCols, MaxRows};

impl<Mods: OnDropBuffer + 'static, T> Max<T> for OpenCL<Mods>
where
    T: Number,
{
    #[inline]
    fn max(&self, x: &Buffer<T, Self>) -> T {
        cpu_exec_reduce_may_unified(self, x, |cpu, x| cpu.max(x))
    }
}

impl<Mods, T> MaxRows<T> for OpenCL<Mods>
where
    T: Number,
    Mods: OnDropBuffer + UnifiedMemChain<Self> + Retrieve<Self, T> + 'static,
{
    #[inline]
    fn max_rows(&self, cols: usize, x: &Buffer<T, Self>) -> Buffer<T, Self> {
        cpu_exec_unary_may_unified(self, x, |cpu, x| cpu.max_rows(cols, x)).unwrap()
    }
}

impl<Mods, T> MaxCols<T> for OpenCL<Mods>
where
    T: Number,
    Mods: OnDropBuffer + UnifiedMemChain<Self> + Retrieve<Self, T> + 'static,
{
    #[inline]
    fn max_cols(&self, rows: usize, cols: usize, x: &Buffer<T, Self>) -> Buffer<T, Self> {
        cpu_exec_unary_may_unified(self, x, |cpu, x| cpu.max_cols(rows, cols, x)).unwrap()
    }
}
