use std::{iter::Sum, ops::AddAssign};

use custos::{
    exec_on_cpu::cpu_exec_reduce_may_unified, prelude::cpu_exec_unary_may_unified, Buffer,
    CDatatype, OpenCL,
};

use crate::{SumCols, SumRows};

impl<T> crate::Sum<T> for OpenCL
where
    T: Default + Copy + Sum,
{
    #[inline]
    fn sum(&self, x: &Buffer<T, Self>) -> T {
        cpu_exec_reduce_may_unified(self, x, |cpu, x| cpu.sum(x))
    }
}

impl<T> SumRows<T> for OpenCL
where
    T: Default + Copy + Sum + AddAssign,
{
    #[inline]
    fn sum_rows(&self, cols: usize, x: &Buffer<T, Self>) -> Buffer<T, Self> {
        cpu_exec_unary_may_unified(self, x, |cpu, x| cpu.sum_rows(cols, x)).unwrap()
    }
}

impl<T> SumCols<T> for OpenCL
where
    T: Default + Copy + Sum,
{
    #[inline]
    fn sum_cols(&self, cols: usize, x: &Buffer<T, Self>) -> Buffer<T, Self> {
        cpu_exec_unary_may_unified(self, x, |cpu, x| cpu.sum_cols(cols, x)).unwrap()
    }
}

pub fn cl_sum<T>(x: &Buffer<T, OpenCL>)
where
    T: CDatatype,
{
}
