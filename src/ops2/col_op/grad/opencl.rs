use custos::{CDatatype, OpenCL, Resolve, ToMarker, MayToCLSource};
use custos::prelude::CLBuffer;

pub fn cl_col_op_grad_lhs<T: CDatatype + Default, LhsGrad>(
    device: &OpenCL,
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
        __kernel void col_op_grad_lhs(__global const {dtype}* lhs, __global const {dtype}* rhs, __global {dtype}* lhs_grad, __global const {dtype}* out_grad, const size_t cols) {{
            size_t idx = get_global_id(0);
            size_t rhs_idx = idx / cols;

            lhs_grad[idx] += {op} * out_grad[idx];
        }}
    ", dtype=T::as_c_type_str(), op=lhs_grad_fn("lhs[idx]".to_marker(), "rhs[rhs_idx]".to_marker()).to_cl_source());

    device.launch_kernel(&src, [lhs.len(), 0, 0], None, &[lhs, rhs, lhs_grad, out_grad, &cols]).unwrap();
}

pub fn cl_col_op_grad_rhs<T: CDatatype + Default, LhsGrad>(
    device: &OpenCL,
    cols: usize,
    lhs: &CLBuffer<T>,
    rhs: &CLBuffer<T>,
    rhs_grad: &mut CLBuffer<T>,
    out_grad: &CLBuffer<T>,
    rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LhsGrad,
) where
    LhsGrad: MayToCLSource,
{
    let src = format!(r#"
        __kernel void col_op_grad_rhs(__global const {dtype}* lhs, __global const {dtype}* rhs, __global {dtype}* rhs_grad, __global const {dtype}* out_grad, const size_t cols) {{
            size_t idx = get_global_id(0);
            size_t rhs_idx = idx % (cols-1);

            {dtype} val = {op} * out_grad[idx];
            // printf("rhs_idx: %d, %f, ", rhs_idx, val);
            // barrier: atomic_add(&rhs_grad[rhs_idx], val);
            
            rhs_grad[rhs_idx] += val;
        }}
    "#, dtype=T::as_c_type_str(), op=rhs_grad_fn("lhs[idx]".to_marker(), "rhs[rhs_idx]".to_marker()).to_cl_source());
    device.launch_kernel(&src, [lhs.len(), 0, 0], None, &[lhs, rhs, rhs_grad, out_grad, &cols]).unwrap();
}

#[cfg(test)]
mod tests {
    use custos::{OpenCL, Device, Resolve, Combiner};

    use crate::{cl_col_op_grad_lhs, test_utils::roughly_equals, cl_col_op_grad_rhs};

    #[test]
    fn test_cl_col_op_grad_lhs_div() -> custos::Result<()> {
        let device = OpenCL::new(0)?;

        #[rustfmt::skip]
        let lhs = [
            1., 2., 3., 
            4., 5., 6.
        ];

        let lhs = device.buffer(&lhs);
        let mut lhs_grad = device.buffer(lhs.len());

        let rhs = device.buffer([-3., 2.]);

        let out_grad = device.buffer([1.4, 2.5, 3.3, 4., 5., 6.]);

        cl_col_op_grad_lhs(&device, 3, &lhs, &rhs, &mut lhs_grad, &out_grad, |_lhs, rhs| Resolve { val: 1., marker: "1" }.div(rhs));

        roughly_equals(
            lhs_grad.read(),
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
        let device = OpenCL::new(0)?;

        #[rustfmt::skip]
        let lhs = [
            1., 2., 3., 
            4., 5., 6.
        ];

        let lhs = device.buffer(&lhs);

        let rhs = device.buffer([-3., 2.]);
        let mut rhs_grad = device.buffer(rhs.len());

        let out_grad = device.buffer([1.4, 2.5, 3.3, 4., 5., 6.]);

        cl_col_op_grad_rhs(&device, 3, &lhs, &rhs, &mut rhs_grad, &out_grad, |lhs, rhs| lhs.div(rhs.mul(rhs).neg()));

        let gf = |lhs: f32, rhs: f32| lhs / -(rhs * rhs);

        roughly_equals(
            rhs_grad.read(),
            &[
                gf(1., -3.) * 1.4 + gf(2., -3.) * 2.5 + gf(3., -3.) * 3.3,
                gf(4., 2.) * 4. + gf(5., 2.) * 5. + gf(6., 2.) * 6.,
            ],
        );

        Ok(())
    }
}
