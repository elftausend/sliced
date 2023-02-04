use custos::{prelude::enqueue_kernel, Buffer, CDatatype, Eval, OpenCL, Resolve, ToMarker};

use crate::BinaryGrad;

impl<T> BinaryGrad<T> for OpenCL
where
    T: CDatatype,
{
    #[inline]
    fn add_binary_grad<LO, RO>(
        &self,
        lhs: &Buffer<T, OpenCL>,
        rhs: &Buffer<T, OpenCL>,
        lhs_grad: &mut Buffer<T, OpenCL>,
        rhs_grad: &mut Buffer<T, OpenCL>,
        out: &Buffer<T, OpenCL>,
        lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LO,
        rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RO,
    ) where
        LO: Eval<T> + ToString,
        RO: Eval<T> + ToString,
    {
        cl_binary_grad(
            self,
            lhs,
            rhs,
            lhs_grad,
            rhs_grad,
            out,
            lhs_grad_fn,
            rhs_grad_fn,
        )
        .unwrap();
    }
}

pub fn cl_binary_grad<T, LO, RO>(
    device: &OpenCL,
    lhs: &Buffer<T, OpenCL>,
    rhs: &Buffer<T, OpenCL>,
    lhs_grad: &mut Buffer<T, OpenCL>,
    rhs_grad: &mut Buffer<T, OpenCL>,
    out: &Buffer<T, OpenCL>,
    lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LO,
    rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RO,
) -> custos::Result<()>
where
    T: CDatatype,
    LO: ToString,
    RO: ToString,
{
    let src = format!(
        "
        __kernel void binary_grad(
            __global const {ty}* lhs, 
            __global const {ty}* rhs,
            __global {ty}* lhs_grad, 
            __global {ty}* rhs_grad,
            __global const {ty}* out
        ) 
        {{
            size_t id = get_global_id(0);
            lhs_grad[id] += {lhs_grad_op} * out[id];
            rhs_grad[id] += {rhs_grad_op} * out[id];
        }}
    ",
        ty = T::as_c_type_str(),
        lhs_grad_op = lhs_grad_fn("lhs[id]".to_marker(), "rhs[id]".to_marker()).to_string(),
        rhs_grad_op = rhs_grad_fn("lhs[id]".to_marker(), "rhs[id]".to_marker()).to_string()
    );

    enqueue_kernel(
        device,
        &src,
        [lhs.len(), 0, 0],
        None,
        &[lhs, rhs, lhs_grad, rhs_grad, out],
    )
}
