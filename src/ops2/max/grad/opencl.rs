use std::ops::AddAssign;

use custos::{prelude::CLBuffer, CDatatype, OpenCL};

use crate::{MaxColsGrad, MaxRowsGrad};

impl<T> MaxRowsGrad<T> for OpenCL
where
    T: PartialEq + Copy + AddAssign + CDatatype,
{
    #[inline]
    fn max_rows_grad(
        &self,
        cols: usize,
        out: &CLBuffer<T>,
        x: &CLBuffer<T>,
        x_grad: &mut CLBuffer<T>,
        out_grad: &CLBuffer<T>,
    ) {
        cl_max_rows_grad(self, cols, out, x, x_grad, out_grad).unwrap();
    }
}

impl<T> MaxColsGrad<T> for OpenCL
where
    T: PartialEq + Copy + AddAssign + CDatatype,
{
    #[inline]
    fn max_cols_grad(
        &self,
        cols: usize,
        out: &CLBuffer<T>,
        x: &CLBuffer<T>,
        x_grad: &mut CLBuffer<T>,
        out_grad: &CLBuffer<T>,
    ) {
        cl_max_cols_grad(self, cols, out, x, x_grad, out_grad).unwrap();
    }
}

// TODO
pub fn cl_max_rows_grad<T: CDatatype>(
    device: &OpenCL,
    cols: usize,
    out: &CLBuffer<T>,
    x: &CLBuffer<T>,
    x_grad: &mut CLBuffer<T>,
    out_grad: &CLBuffer<T>,
) -> custos::Result<()> {
    let src = format!(
        r#"
        __kernel void max_rows_grad(__global const {dtype} *out, __global const {dtype} *x, __global {dtype} *x_grad, __global const {dtype} *out_grad, int cols) {{
            int col = get_global_id(0);
            int row = get_global_id(1);
            int grad_idx = row * cols + col;

            if (out[col] == x[grad_idx]) {{
                x_grad[grad_idx] += out_grad[col];
            }}
        }}
        "#,
        dtype = T::as_c_type_str()
    );

    device.launch_kernel(
        &src,
        [cols, x.len() / cols, 0],
        None,
        &[out, x, x_grad, out_grad, &(cols as i32)],
    )
}

pub fn cl_max_cols_grad<T: CDatatype>(
    device: &OpenCL,
    cols: usize,
    out: &CLBuffer<T>,
    x: &CLBuffer<T>,
    x_grad: &mut CLBuffer<T>,
    out_grad: &CLBuffer<T>,
) -> custos::Result<()> {
    let src = format!(
        r#"
        __kernel void max_cols_grad(__global const {dtype} *out, __global const {dtype} *x, __global {dtype} *x_grad, __global const {dtype} *out_grad, int cols) {{
            int row = get_global_id(0);
            int col = get_global_id(1);
            int grad_idx = row * cols + col;

            if (out[row] == x[grad_idx]) {{
                x_grad[grad_idx] += out_grad[row];
            }}
        }}
        "#,
        dtype = T::as_c_type_str()
    );

    device.launch_kernel(
        &src,
        [x.len() / cols, cols, 0],
        None,
        &[out, x, x_grad, out_grad, &(cols as i32)],
    )
}

#[cfg(test)]
mod tests {
    use custos::{Device, OpenCL};

    use crate::{cl_max_cols_grad, cl_max_rows_grad, MaxCols, MaxRows};

    #[test]
    fn test_max_rows_grad() -> custos::Result<()> {
        let device = OpenCL::new(0)?;

        #[rustfmt::skip]
        let x = [-3, 2, 3, 1,
                            1, 5, -5, 4,
                            -9, -2, -4, -1];

        let x = device.buffer(x);
        let mut x_grad = device.buffer(x.len());

        let out = device.max_rows(4, &x);

        let out_grad = device.buffer(&[2, 3, 4, 1]);
        cl_max_rows_grad(&device, 4, &out, &x.clone(), &mut x_grad, &out_grad)?;

        #[rustfmt::skip]
        let expected = vec![
            0, 0, 4, 0,
            2, 3, 0, 1,
            0, 0, 0, 0,
        ];

        assert_eq!(expected, x_grad.read());
        Ok(())
    }

    #[test]
    fn test_max_cols_grad() -> custos::Result<()> {
        let device = OpenCL::new(0)?;

        #[rustfmt::skip]
        let x = [-3, 2, 3, 1,
                            1, 5, -5, 4,
                            -9, -2, -4, -1];

        let x = device.buffer(x);
        let mut x_grad = device.buffer(x.len());

        let out = device.max_cols(3, 4, &x);

        let out_grad = device.buffer(&[1, 2, 3]);
        cl_max_cols_grad(&device, 4, &out, &x.clone(), &mut x_grad, &out_grad)?;

        #[rustfmt::skip]
        let expected = vec![
            0, 0, 1, 0,
            0, 2, 0, 0,
            0, 0, 0, 3,
        ];

        assert_eq!(expected, x_grad.read());
        Ok(())
    }
}
