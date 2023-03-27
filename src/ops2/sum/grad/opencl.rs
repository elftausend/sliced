use std::ops::AddAssign;

use custos::{prelude::CLBuffer, CacheReturn, Device, OpenCL, Shape, CDatatype};

use crate::SumRowsGrad;

// TODO
pub fn cl_sum_rows_grad<T: CDatatype>(
    device: &OpenCL,
    cols: usize,
    x_grad: &mut CLBuffer<T>,
    out_grad: &CLBuffer<T>,
) -> custos::Result<()> {
    let src = format!("
        __kernel void sum_rows_grad(__global {dtype}* x_grad, __global const {dtype}* out_grad, int cols) {{
            size_t row = get_global_id(0);

            size_t start_idx = row * cols;

            for (size_t col = start_idx; col < cols; col++ ) {{
                size_t grad_idx = start_idx + col;
                x_grad[grad_idx] += out_grad[col];
            }}
            // TODO: add BARRIER
    
        }}
    ", dtype=T::as_c_type_str());
    
    device.launch_kernel(&src, [x_grad.len() / cols, 0, 0], None, &[x_grad, out_grad, &(cols as i32)])
}

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

#[cfg(test)]
mod tests {
    use custos::{OpenCL, Device, Buffer};

    use crate::cl_sum_rows_grad;


    #[test]
    fn test_cl_sum_rows() -> custos::Result<()> {
        #[rustfmt::skip]
        let _x = [
            1, 2, 3,
            3, 2, 1,
            2, 3, 1,
            4, 8, 1
        ];

        let device = OpenCL::new(0)?;

        let mut x_grad = device.buffer(12);

        let out_grad = device.buffer([2, 4, -1]);

        cl_sum_rows_grad(&device, 3, &mut x_grad, &out_grad)?;

        println!("x_grad: {x_grad:?}");

        Ok(())
    }
}
