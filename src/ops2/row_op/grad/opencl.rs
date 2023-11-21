use std::ops::{AddAssign, Mul};

use crate::{slice_add_row_op_grad, RowOpGrad};
use custos::{
    opencl::CLBuffer,
    prelude::{cpu_exec_binary_may_unified_mut, One},
    Base, Buffer, OpenCL, Retrieve,
};

impl<T, Mods: Retrieve<Self, T> + 'static> RowOpGrad<T> for OpenCL<Mods>
where
    T: Copy + Default + AddAssign + One + Mul<Output = T>,
{
    fn row_op_grad(
        &self,
        cols: usize,
        lhs: &Buffer<T, Self>,
        rhs: &Buffer<T, Self>,
        lhs_grad: &mut Buffer<T, Self>,
        rhs_grad: &mut Buffer<T, Self>,
        out_grad: &Buffer<T, Self>,
        lhs_grad_fn: impl Fn(T) -> T,
        rhs_grad_fn: impl Fn(T) -> T,
    ) {
        use custos::{Buffer, WriteBuf, CPU};

        #[rustfmt::skip]
        custos::cl_cpu_exec_unified_mut!(
            self,
            lhs, rhs, out_grad
            WRITE_TO<
                lhs_grad, lhs_grad_cpu,
                rhs_grad, rhs_grad_cpu
            >
            self.cpu.row_op_grad(cols, &lhs, &rhs,
                &mut lhs_grad_cpu, &mut rhs_grad_cpu,
                &out_grad, lhs_grad_fn, rhs_grad_fn
            )
        );
    }

    fn add_row_grad(
        &self,
        rows: usize,
        cols: usize,
        lhs_grad: &mut Buffer<T, Self>,
        rhs_grad: &mut Buffer<T, Self>,
        out_grad: &Buffer<T, Self>,
    ) {
        use custos::{Buffer, WriteBuf, CPU};

        #[rustfmt::skip]
        custos::cl_cpu_exec_unified_mut!(
            self,
            out_grad
            WRITE_TO<
                lhs_grad, lhs_grad_cpu,
                rhs_grad, rhs_grad_cpu
            >
            slice_add_row_op_grad(rows, cols, &mut lhs_grad_cpu, &mut rhs_grad_cpu, &out_grad)
            // self.cpu.add_row_grad(rows, cols, &mut lhs_grad_cpu, &mut rhs_grad_cpu, &out_grad)
        );
    }

    #[inline]
    fn add_row_mut_grad(
        &self,
        rows: usize,
        cols: usize,
        rhs_grad: &mut Buffer<T, Self>,
        out_grad: &Buffer<T, Self>,
    ) {
        cpu_exec_binary_may_unified_mut(self, rhs_grad, out_grad, |cpu, rhs_grad, out_grad| {
            cpu.add_row_mut_grad(rows, cols, rhs_grad, out_grad)
        })
        .unwrap();
    }
}

pub fn try_cl_row_op_grad_rhs<T: CDatatype + Default, O: MayToCLSource>(
    device: &CLDevice,
    lhs: &CLPtr<T>,
    rhs: &CLPtr<T>,
    out: &mut CLPtr<T>,
    //lhs_grad: &mut Buffer<T, Self>,
    rhs_grad: &mut Buffer<T, Self>,
    out_grad: &Buffer<T, Self>,
    //lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> O,
    rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> O,
    rows: usize,
    cols: usize,
    //op: impl Fn(Resolve<T>, Resolve<T>) -> O,
) -> custos::Result<()> {
    let src = format!("
        __kernel void cl_rop_op(__global const {dtype}* lhs, global const {dtype}* rhs, global {dtype}* out_grad, int rows, int cols) {{
            size_t c = get_global_id(0);

            if (c >= cols) {{
                return;
            }}
            
            {dtype} rhs_val = rhs[c];
            {dtype} acc = 0;
            for (int r = 0; r < rows; r++) {{
                {dtype} lhs_val = lhs[r * cols + c];
                acc += {op} * out_grad[r * cols + c];
            }}
            
            rhs_grad[c] = acc;

        }}
    ", dtype = T::C_DTYPE_STR, op = op("lhs_val".to_marker(), "rhs_val".to_marker()).to_cl_source());

    let gws = [((rows + 32) / 32) * 32, ((cols + 32) / 32) * 32, 0];
    enqueue_kernel(device, &src, gws, Some([32, 4, 0]), &[lhs, rhs, out, &rows, &cols])
}
