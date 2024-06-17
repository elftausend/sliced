use std::ops::AddAssign;

use custos::{
    opencl::{CLDevice, KernelLaunch},
    prelude::CLBuffer,
    CDatatype, OnDropBuffer, OpenCL, Shape,
};

use crate::SumRowsGrad;

// measured performance
pub fn cl_sum_rows_grad<T: CDatatype>(
    device: &CLDevice,
    cols: usize,
    x_grad: &mut CLBuffer<T>,
    out_grad: &CLBuffer<T>,
) -> custos::Result<()> {
    let src = format!("
        __kernel void sum_rows_grad(__global {dtype}* x_grad, __global const {dtype}* out_grad, int cols) {{
            size_t row = get_global_id(0);

            size_t start_idx = row * cols;

            for (size_t col = 0; col < cols; col++ ) {{
                size_t grad_idx = start_idx + col;
                x_grad[grad_idx] += out_grad[col];
            }}    
        }}
    ", dtype=T::C_DTYPE_STR);

    device.launch_kernel(
        &src,
        [x_grad.len() / cols, 0, 0],
        None,
        &[x_grad, out_grad, &(cols as i32)],
    )
}

pub fn cl_sum_cols_grad<T: CDatatype>(
    device: &CLDevice,
    cols: usize,
    x_grad: &mut CLBuffer<T>,
    out_grad: &CLBuffer<T>,
) -> custos::Result<()> {
    let src = format!("
        __kernel void sum_cols_grad(__global {dtype}* x_grad, __global const {dtype}* out_grad, int cols) {{
            size_t row = get_global_id(0);

            size_t start_idx = row * cols;

            for (size_t col = 0; col < cols; col++ ) {{
                size_t grad_idx = start_idx + col;
                x_grad[grad_idx] += out_grad[row];
            }}    
        }}
    ", dtype=T::C_DTYPE_STR);

    device.launch_kernel(
        &src,
        [x_grad.len() / cols, 0, 0],
        None,
        &[x_grad, out_grad, &(cols as i32)],
    )
}

pub fn cl_sum_rows_grad_modulo<T: CDatatype>(
    device: &CLDevice,
    cols: usize,
    x_grad: &mut CLBuffer<T>,
    out_grad: &CLBuffer<T>,
) -> custos::Result<()> {
    let src = format!("
        __kernel void sum_rows_grad_modulo(__global {dtype}* x_grad, __global const {dtype}* out_grad, int cols) {{
            size_t idx = get_global_id(0);

            size_t out_grad_idx = idx % cols;

            x_grad[idx] += out_grad[out_grad_idx];
        }}
    ", dtype=T::C_DTYPE_STR);

    device.launch_kernel(
        &src,
        [x_grad.len(), 0, 0],
        None,
        &[x_grad, out_grad, &(cols as i32)],
    )
}

impl<T, IS, OS, Mods: OnDropBuffer + 'static> SumRowsGrad<T, IS, OS> for OpenCL<Mods>
where
    T: Default + Copy + AddAssign + 'static,
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
        use custos::{Base, Buffer, WriteBuf, CPU};
        // let cpu = custos::CPU::<custos::Autograd<custos::Base>>::new();

        #[rustfmt::skip]
        custos::cl_cpu_exec_unified_mut!(
            self,
            out_grad;
            WRITE_TO<x_grad, x_grad_cpu>
            self.cpu.sum_rows_grad(cols, &mut x_grad_cpu, &out_grad)
        );
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use custos::{Buffer, Device, OpenCL};

    use crate::{cl_sum_cols_grad, cl_sum_rows_grad, cl_sum_rows_grad_modulo};

    #[test]
    fn test_cl_sum_cols_grad() -> custos::Result<()> {
        #[rustfmt::skip]
        let _x = [
            1, 2, 3,
            3, 2, 1,
            2, 3, 1,
            4, 8, 1
        ];

        let device = OpenCL::<custos::Autograd<custos::Base>>::new(0)?;

        let mut x_grad = device.buffer::<_, (), _>(12);
        let out_grad = device.buffer([2, 3, 4, -1]);

        cl_sum_cols_grad(&device, 3, &mut x_grad, &out_grad)?;

        #[rustfmt::skip]
        #[rustfmt::skip]
        let expected = [
            2, 2, 2, 
            3, 3, 3, 
            4, 4, 4, 
            -1, -1, -1
        ];

        assert_eq!(x_grad.read(), expected);
        Ok(())
    }

    #[test]
    fn test_cl_sum_rows_grad() -> custos::Result<()> {
        #[rustfmt::skip]
        let _x = [
            1, 2, 3,
            3, 2, 1,
            2, 3, 1,
            4, 8, 1
        ];

        let device = OpenCL::<custos::Autograd<custos::Base>>::new(0)?;

        let mut x_grad = device.buffer::<_, (), _>(12);
        let out_grad = device.buffer([2, 4, -1]);

        cl_sum_rows_grad(&device, 3, &mut x_grad, &out_grad)?;

        #[rustfmt::skip]
        let expected = [
            2, 4, -1, 
            2, 4, -1, 
            2, 4, -1, 
            2, 4, -1
        ];
        assert_eq!(x_grad.read(), expected);
        Ok(())
    }

    #[test]
    fn test_cl_sum_rows_modulo() -> custos::Result<()> {
        #[rustfmt::skip]
        let _x = [
            1, 2, 3,
            3, 2, 1,
            2, 3, 1,
            4, 8, 1
        ];

        let device = OpenCL::<custos::Autograd<custos::Base>>::new(0)?;

        let mut x_grad = device.buffer::<_, (), _>(12);

        let out_grad = device.buffer([2, 4, -1]);

        cl_sum_rows_grad_modulo(&device, 3, &mut x_grad, &out_grad)?;

        #[rustfmt::skip]
        let expected = [
            2, 4, -1, 
            2, 4, -1, 
            2, 4, -1, 
            2, 4, -1
        ];
        assert_eq!(x_grad.read(), expected);

        Ok(())
    }

    #[test]
    fn test_cl_sum_rows_modulo_vs_for() -> custos::Result<()> {
        let device = OpenCL::<custos::Autograd<custos::Base>>::new(0)?;

        let mut x_grad = device.buffer::<_, (), _>(4000 * 1000);

        let out_grad: Buffer<i32, _> = device.buffer(0..1000);

        const TIMES: usize = 1000;

        let start = Instant::now();

        for _ in 0..TIMES {
            cl_sum_rows_grad_modulo(&device, out_grad.len(), &mut x_grad, &out_grad)?;
        }
        println!("elapsed (modulo): {:?}", start.elapsed());

        Ok(())
    }

    #[test]
    fn test_cl_sum_rows_for_large() -> custos::Result<()> {
        let device = OpenCL::<custos::Autograd<custos::Base>>::new(0)?;

        let mut x_grad = device.buffer::<_, (), _>(4000 * 1000);

        let out_grad: Buffer<i32, _> = device.buffer(0..1000);

        const TIMES: usize = 1000;

        let start = Instant::now();

        for _ in 0..TIMES {
            cl_sum_rows_grad(&device, out_grad.len(), &mut x_grad, &out_grad)?;
        }

        println!("elapsed (for): {:?}", start.elapsed());

        Ok(())
    }
}
