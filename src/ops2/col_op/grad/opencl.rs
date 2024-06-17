use custos::{
    opencl::{CLDevice, KernelLaunch},
    prelude::CLBuffer,
};
use custos::{CDatatype, MayToCLSource, OpenCL, Resolve, ToMarker};

pub fn cl_col_op_grad_lhs<T: CDatatype + Default, LhsGrad>(
    device: &CLDevice,
    cols: usize,
    lhs: &CLBuffer<T>,
    rhs: &CLBuffer<T>,
    lhs_grad: &mut CLBuffer<T>,
    out_grad: &CLBuffer<T>,
    lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LhsGrad,
) where
    LhsGrad: MayToCLSource,
{
    let src = format!("
        __kernel void col_op_grad_lhs(__global const {dtype}* lhs, __global const {dtype}* rhs, __global {dtype}* lhs_grad, __global const {dtype}* out_grad, long cols) {{
            size_t idx = get_global_id(0);
            size_t rhs_idx = idx / cols;

            lhs_grad[idx] += {op} * out_grad[idx];
        }}
    ", dtype=T::C_DTYPE_STR, op=lhs_grad_fn("lhs[idx]".to_marker(), "rhs[rhs_idx]".to_marker()).to_cl_source());

    device
        .launch_kernel(
            &src,
            [lhs.len(), 0, 0],
            None,
            &[lhs, rhs, lhs_grad, out_grad, &cols],
        )
        .unwrap();
}

pub fn cl_col_op_grad_rhs<T: CDatatype + Default, LhsGrad>(
    device: &CLDevice,
    cols: usize,
    lhs: &CLBuffer<T>,
    rhs: &CLBuffer<T>,
    rhs_grad: &mut CLBuffer<T>,
    out_grad: &CLBuffer<T>,
    rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LhsGrad,
) where
    LhsGrad: MayToCLSource,
{
    let src = format!(
        r#"
        __kernel void col_op_grad_rhs(__global const {dtype}* lhs, __global const {dtype}* rhs, __global {dtype}* rhs_grad, __global const {dtype}* out_grad, long cols) {{
            size_t idx = get_global_id(0);

            {dtype} sum = 0;
            for (int col = 0; col < cols; col++) {{
                {dtype} lhs_val = lhs[idx * cols + col];
                sum += {op} * out_grad[idx * cols + col];
            }}

            rhs_grad[idx] += sum;
        }}
    "#,
        dtype = T::C_DTYPE_STR,
        op = rhs_grad_fn("lhs_val".to_marker(), "rhs[idx]".to_marker()).to_cl_source()
    );
    device
        .launch_kernel(
            &src,
            [rhs.len(), 0, 0],
            None,
            &[lhs, rhs, rhs_grad, out_grad, &cols],
        )
        .unwrap();
}

#[cfg(test)]
mod tests {
    use custos::{Combiner, Device, OpenCL, Resolve};

    use crate::{cl_col_op_grad_lhs, cl_col_op_grad_rhs, test_utils::roughly_equals};

    #[test]
    fn test_cl_col_op_grad_lhs_div() -> custos::Result<()> {
        let device = OpenCL::<custos::Autograd<custos::Base>>::new(0)?;

        #[rustfmt::skip]
        let lhs = [
            1f32, 2., 3., 
            4., 5., 6.
        ];

        let lhs = device.buffer(&lhs);
        let mut lhs_grad = device.buffer::<_, (), _>(lhs.len());

        let rhs = device.buffer([-3f32, 2.]);

        let out_grad = device.buffer([1.4f32, 2.5, 3.3, 4., 5., 6.]);

        cl_col_op_grad_lhs(
            &device,
            3,
            &lhs,
            &rhs,
            &mut lhs_grad,
            &out_grad,
            |_lhs, rhs| {
                Resolve {
                    val: 1.,
                    marker: "1",
                }
                .div(rhs)
            },
        );

        roughly_equals(
            &lhs_grad.read_to_vec(),
            &[
                1.4 * 1. / -3.,
                2.5 * 1. / -3.,
                3.3 * 1. / -3.,
                4. * 1. / 2.,
                5. * 1. / 2.,
                6. * 1. / 2.,
            ],
        );

        Ok(())
    }

    #[test]
    fn test_cl_col_op_grad_rhs_div() -> custos::Result<()> {
        let device = OpenCL::<custos::Autograd<custos::Base>>::new(0)?;

        #[rustfmt::skip]
        let lhs = [
            1., 2., 3., 
            4., 5., 6.
        ];

        let lhs = device.buffer(&lhs);

        let rhs = device.buffer([-3., 2.]);
        let mut rhs_grad = device.buffer::<_, (), _>(rhs.len());

        let out_grad = device.buffer([1.4, 2.5, 3.3, 4., 5., 6.]);

        cl_col_op_grad_rhs(
            &device,
            3,
            &lhs,
            &rhs,
            &mut rhs_grad,
            &out_grad,
            |lhs, rhs| lhs.div(rhs.mul(rhs).neg()),
        );

        let gf = |lhs: f32, rhs: f32| lhs / -(rhs * rhs);

        roughly_equals(
            &rhs_grad.read_to_vec(),
            &[
                gf(1., -3.) * 1.4 + gf(2., -3.) * 2.5 + gf(3., -3.) * 3.3,
                gf(4., 2.) * 4. + gf(5., 2.) * 5. + gf(6., 2.) * 6.,
            ],
        );

        Ok(())
    }
}
