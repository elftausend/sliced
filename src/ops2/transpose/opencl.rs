use custos::{
    opencl::{CLDevice, CLPtr},
    Buffer, CDatatype, OpenCL, Retrieve, Retriever,
};

use crate::{
    assign_or_set::{AssignOrSet, Set},
    Transpose,
};

impl<Mods: Retrieve<Self, T>, T: CDatatype> Transpose<T> for OpenCL<Mods> {
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, Self>) -> Buffer<T, Self> {
        let mut out = self.retrieve(x.len(), x).unwrap();
        cl_transpose::<T, Set>(self, x, &mut out, rows, cols).unwrap();
        out
    }
}

pub fn cl_transpose<T: CDatatype, AOS: AssignOrSet<T>>(
    device: &CLDevice,
    x: &CLPtr<T>,
    out: &mut CLPtr<T>,
    rows: usize,
    cols: usize,
) -> custos::Result<()> {
    use custos::opencl::enqueue_kernel;

    let src = format!(
        "
        #define MODULO(x,N) (x % N)
        #define I0 {rows}
        #define I1 {cols}
        #define I_idx(i0,i1) ((size_t)(i0))*I1+(i1)
        #define I_idx_mod(i0,i1) MODULO( ((size_t)(i0)) ,I0)*I1+MODULO( (i1),I1)

        #define MODULO(x,N) (x % N)
        #define O0 {cols}
        #define O1 {rows}
        #define O_idx(o0,o1) ((size_t)(o0))*O1+(o1)
        #define O_idx_mod(o0,o1) MODULO( ((size_t)(o0)) ,O0)*O1+MODULO( (o1),O1)
        __kernel void transpose(__global const {datatype}* I, __global {datatype}* O) {{
            size_t gid = get_global_id(0);
            size_t gid_original = gid;size_t i1 = gid % I1;size_t i0 = gid / I1;gid = gid_original;
        
            O[O_idx(i1,i0)] {aos} I[gid];
        }}
    
   ",
        aos = AOS::STR_OP,
        datatype = T::C_DTYPE_STR
    );

    let gws = [x.len(), 0, 0];
    enqueue_kernel(device, &src, gws, None, &[x, out])?;
    Ok(())
}
