use custos::{
    prelude::enqueue_kernel, Buffer, CDatatype, Device, Eval, OpenCL, Resolve, Shape, ToMarker,
};

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

pub fn cl_binary_ew<T, S, O>(
    device: &OpenCL,
    lhs: &Buffer<T, OpenCL, S>,
    rhs: &Buffer<T, OpenCL, S>,
    out: &mut Buffer<T, OpenCL, S>,
    f: impl Fn(Resolve<T>, Resolve<T>) -> O,
) -> custos::Result<()>
where
    S: Shape,
    T: CDatatype,
    O: ToString,
{
    let src = format!(
        "
        __kernel void binary_ew(__global const {ty}* lhs, __global const {ty}* rhs, __global {ty}* out) {{
            size_t id = get_global_id(0);
            out[id] = {op};
        }}
    ",
        ty = T::as_c_type_str(),
        op = f("lhs[id]".to_marker(), "rhs[id]".to_marker()).to_string()
    );

    enqueue_kernel(device, &src, [lhs.len(), 0, 0], None, &[lhs, rhs, out])
}

#[cfg(test)]
mod tests {
    use custos::{Buffer, Combiner, OpenCL};

    use super::cl_binary_ew;

    #[test]
    fn test_binary_ew() -> custos::Result<()> {
        let device = OpenCL::new(0)?;

        let lhs = Buffer::from((&device, &[1, 5, 3, 2, 6]));
        let rhs = Buffer::from((&device, &[-1, 2, 9, 1, -2]));

        let mut out = Buffer::new(&device, 5);

        cl_binary_ew(&device, &lhs, &rhs, &mut out, |a, b| a.add(b))?;

        assert_eq!(out.read(), vec![0, 7, 12, 3, 4]);

        Ok(())
    }
}
