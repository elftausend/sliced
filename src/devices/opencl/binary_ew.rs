use custos::{prelude::enqueue_kernel, Buffer, CDatatype, Device, Eval, OpenCL, Resolve, ToMarker};

use crate::BinaryElementWise;

impl<T> BinaryElementWise<T> for OpenCL
where
    T: CDatatype,
{
    #[inline]
    fn binary_ew<O>(
        &self,
        lhs: &Buffer<T, Self>,
        rhs: &Buffer<T, Self>,
        f: impl Fn(Resolve<T>, Resolve<T>) -> O,
    ) -> Buffer<T, Self>
    where
        O: Eval<T> + ToString,
    {
        let mut out = self.retrieve(lhs.len());
        cl_binary_ew(self, lhs, rhs, &mut out, f).unwrap();
        out
    }
}

pub fn cl_binary_ew<T, O>(
    device: &OpenCL,
    lhs: &Buffer<T, OpenCL>,
    rhs: &Buffer<T, OpenCL>,
    out: &mut Buffer<T, OpenCL>,
    f: impl Fn(Resolve<T>, Resolve<T>) -> O,
) -> custos::Result<()>
where
    T: CDatatype,
    O: Eval<T> + ToString,
{
    let src = format!(
        "
        __kernel void binary_ew(const {ty}* lhs, const {ty}* rhs, {ty}* out) {{
            size_t id = get_global_id(0);
            out[id] = {op};
        }}
    ",
        ty = T::as_c_type_str(),
        op = f("lhs[id]".to_marker(), "rhs[id]".to_marker()).to_string()
    );

    enqueue_kernel(device, &src, [lhs.len(), 0, 0], None, &[lhs, rhs, out])
}
