use custos::{
    opencl::{CLDevice, CLPtr}, prelude::enqueue_kernel, Buffer, CDatatype, Eval, MayToCLSource, OnDropBuffer, OpenCL, Resolve, Shape, ToMarker
};

use super::BinaryElementWiseGrad;

impl<T, Mods: OnDropBuffer, S: Shape> BinaryElementWiseGrad<T, S> for OpenCL<Mods>
where
    T: CDatatype + Default,
{
    #[inline]
    fn binary_ew_grad<LO, RO>(
        &self,
        lhs: &Buffer<T, Self, S>,
        rhs: &Buffer<T, Self, S>,
        lhs_grad: &mut Buffer<T, Self, S>,
        rhs_grad: &mut Buffer<T, Self, S>,
        out: &Buffer<T, Self, S>,
        lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LO,
        rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RO,
    ) where
        LO: Eval<T> + MayToCLSource,
        RO: Eval<T> + MayToCLSource,
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
    device: &CLDevice,
    lhs: &CLPtr<T>,
    rhs: &CLPtr<T>,
    lhs_grad: &mut CLPtr<T>,
    rhs_grad: &mut CLPtr<T>,
    out: &CLPtr<T>,
    lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LO,
    rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RO,
) -> custos::Result<()>
where
    T: CDatatype + Default,
    LO: MayToCLSource,
    RO: MayToCLSource,
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
        ty = T::C_DTYPE_STR,
        lhs_grad_op = lhs_grad_fn("lhs[id]".to_marker(), "rhs[id]".to_marker()).to_cl_source(),
        rhs_grad_op = rhs_grad_fn("lhs[id]".to_marker(), "rhs[id]".to_marker()).to_cl_source()
    );

    enqueue_kernel(
        device,
        &src,
        [lhs.len(), 0, 0],
        None,
        &[lhs, rhs, lhs_grad, rhs_grad, out],
    )
}
