use custos::{
    exec_on_cpu::cpu_exec_reduce_may_unified, prelude::{cpu_exec_unary_may_unified, Number}, Buffer, OpenCL,
};

use crate::{Max, MaxCols, MaxRows};

impl<T> Max<T> for OpenCL
where
    T: Number,
{
    #[inline]
    fn max(&self, x: &Buffer<T, Self>) -> T {
        cpu_exec_reduce_may_unified(self, x, |cpu, x| cpu.max(x))
    }
}

impl<T> MaxRows<T> for OpenCL
where
    T: Number,
{
    #[inline]
    fn max_rows(&self, cols: usize, x: &Buffer<T, Self>) -> Buffer<T, Self> {
        cpu_exec_unary_may_unified(self, x, |cpu, x| cpu.max_rows(cols, x)).unwrap()
    }
}

impl<T> MaxCols<T> for OpenCL
where
    T: Number,
{
    #[inline]
    fn max_cols(&self, rows: usize, cols: usize, x: &Buffer<T, Self>) -> Buffer<T, Self> {
        cpu_exec_unary_may_unified(self, x, |cpu, x| cpu.max_cols(rows, cols, x)).unwrap()
    }
}
