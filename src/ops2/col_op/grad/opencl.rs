use custos::{prelude::CLBuffer, CDatatype, OpenCL, Resolve};

pub fn cl_col_op_grad_lhs<T: CDatatype, LhsGrad>(
    device: &OpenCL,
    cols: usize,
    lhs: &CLBuffer<T>,
    rhs: &CLBuffer<T>,
    lhs_grad: &mut CLBuffer<T>,
    out_grad: &CLBuffer<T>,
    lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LhsGrad,
    // rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RhsGrad,
) 
where
    LhsGrad: ToString
{
    let src = format!("
        __kernel void col_op_grad_lhs() {{
            
        }}
    ");
}
