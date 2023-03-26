use std::ops::AddAssign;

use custos::{OpenCL, Shape, CacheReturn};

use crate::SumRowsGrad;

// TODO
fn x() {}

impl<T, IS, OS> SumRowsGrad<T, IS, OS> for OpenCL
where
    T: Default + Copy + AddAssign,
    IS: Shape,
    OS: Shape,
{
    #[inline]
    fn sum_rows_grad(
        &self,
        cols: usize,
        x_grad: &mut custos::Buffer<T, Self, IS>,
        out_grad: &custos::Buffer<T, Self, OS>,
    ) {
        use custos::{Buffer, WriteBuf, CPU};
        let cpu = custos::CPU::new();

        #[rustfmt::skip]
        custos::cl_cpu_exec_unified_mut!(
            self,
            out_grad
            WRITE_TO<x_grad, x_grad_cpu>
            self.cpu.sum_rows_grad(cols, &mut x_grad_cpu, &out_grad)
        );
    }
}
