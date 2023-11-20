use std::ops::{Add, AddAssign};

use custos::{
    opencl::{enqueue_kernel, CLDevice, CLPtr},
    Buffer, CDatatype, MayToCLSource, OpenCL, Resolve, ToMarker, Combiner, OnDropBuffer, Retrieve, Retriever,
};

use crate::RowOp;

impl<T, Mods: OnDropBuffer + Retrieve<Self, T>> RowOp<T> for OpenCL<Mods>
where
    T: Copy + Default + Add<Output = T> + AddAssign + CDatatype + 'static,
{
    #[inline]
    fn row_op<O: MayToCLSource>(
        &self,
        cols: usize,
        lhs: &Buffer<T, Self>,
        rhs: &Buffer<T, Self>,
        f: impl Fn(Resolve<T>, Resolve<T>) -> O,
    ) -> Buffer<T, Self> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));
        
        try_cl_row_op(self, lhs, rhs, &mut out, lhs.len / cols, cols, f).unwrap();
        out
    }

    #[inline]
    fn add_row_mut(
        &self,
        rows: usize,
        cols: usize,
        lhs: &mut Buffer<T, Self>,
        rhs: &Buffer<T, Self>,
    ) {
        try_cl_row_op(self, unsafe { &lhs.shallow() } , rhs, lhs, rows, cols, |lhs, rhs| lhs.add(rhs)).unwrap();
        // cpu_exec_binary_may_unified_mut(self, lhs, rhs, |cpu, lhs, rhs| {
        //     cpu.add_row_mut(rows, cols, lhs, rhs)
        // })
        // .unwrap()
    }
}

pub fn try_cl_row_op<T: CDatatype + Default, O: MayToCLSource>(
    device: &CLDevice,
    lhs: &CLPtr<T>,
    rhs: &CLPtr<T>,
    out: &mut CLPtr<T>,
    rows: usize,
    cols: usize,
    op: impl Fn(Resolve<T>, Resolve<T>) -> O,
) -> custos::Result<()> {
    let src = format!("
        __kernel void cl_rop_op(__global const {dtype}* lhs, global const {dtype}* rhs, global {dtype}* out, int rows, int cols) {{
            size_t r = get_global_id(0);
            size_t c = get_global_id(1);

            if (r >= rows || c >= cols) {{
                return;
            }}

            {dtype} lhs_val = lhs[r * cols + c];
            {dtype} rhs_val = rhs[c];
            out[r * cols + c] = {op};

        }}
    ", dtype = T::C_DTYPE_STR, op = op("lhs_val".to_marker(), "rhs_val".to_marker()).to_cl_source());

    let gws = [((rows + 32) / 32) * 32, ((cols + 32) / 32) * 32, 0];
    enqueue_kernel(device, &src, gws, Some([32, 4, 0]), &[lhs, rhs, out, &rows, &cols])
}

#[cfg(test)]
mod tests {
    use custos::{Base, Device, OpenCL, Combiner};

    use crate::try_cl_row_op;

    #[test]
    fn test_cl_row_op() {
        let device = OpenCL::<Base>::new(0).unwrap();

        #[rustfmt::skip]
        let lhs = device.buffer([
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ]);

        let rhs = device.buffer([1, 2, 3]);
        let mut out = device.buffer::<i32, (), _>(lhs.len());

        try_cl_row_op(&device, &lhs, &rhs, &mut out, 3, 3, |lhs, rhs| lhs.add(rhs)).unwrap();

        assert_eq!(out.read_to_vec(), [2, 4, 6, 5, 7, 9, 8, 10, 12]);
    }
}
