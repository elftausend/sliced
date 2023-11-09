use custos::{
    prelude::enqueue_kernel, Buffer, CDatatype, Device, Eval, MayToCLSource, OpenCL, Resolve,
    Shape, ToMarker, Retriever, UnifiedMemChain
};

use super::BinaryElementWise;

impl<T, S: Shape> BinaryElementWise<T, S> for OpenCL
where
    T: CDatatype + Default,
{
    #[inline]
    fn binary_ew<O>(
        &self,
        lhs: &Buffer<T, Self, S>,
        rhs: &Buffer<T, Self, S>,
        f: impl Fn(Resolve<T>, Resolve<T>) -> O,
    ) -> Buffer<T, Self, S>
    where
        O: Eval<T> + MayToCLSource,
    {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));
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
    T: CDatatype + Default,
    O: MayToCLSource,
{
    let src = format!(
        "
        __kernel void binary_ew(__global const {ty}* lhs, __global const {ty}* rhs, __global {ty}* out) {{
            size_t id = get_global_id(0);
            out[id] = {op};
        }}
    ",
        ty = T::C_DTYPE_STR,
        op = f("lhs[id]".to_marker(), "rhs[id]".to_marker()).to_cl_source()
    );

    enqueue_kernel(device, &src, [lhs.len(), 0, 0], None, &[lhs, rhs, out])
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use custos::{Buffer, Combiner, OpenCL};

    use super::cl_binary_ew;

    #[test]
    fn test_binary_ew() -> custos::Result<()> {
        let device = OpenCL::<custos::Base>::new(0)?;

        let lhs = Buffer::from((&device, &[1, 5, 3, 2, 6]));
        let rhs = Buffer::from((&device, &[-1, 2, 9, 1, -2]));

        let mut out = Buffer::new(&device, 5);

        cl_binary_ew(&device, &lhs, &rhs, &mut out, |a, b| a.add(b))?;

        assert_eq!(out.read(), vec![0, 7, 12, 3, 4]);

        Ok(())
    }

    #[test]
    fn test_cpu_exec_macro() -> custos::Result<()> {
        use crate::{BinaryElementWise, Buffer, CPU, custos::Base};

        let device = crate::OpenCL::<custos::Base>::new(0)?;

        let cpu = CPU::<custos::Base>::new();

        let lhs = Buffer::from((&device, [1, 2, 3]));
        let rhs = Buffer::from((&device, [1, 2, 3]));

        let a = custos::cpu_exec!(
            device, cpu, lhs, rhs; cpu.add(&lhs, &rhs)
        );
        //let a = Buffer::from((&device, a));

        assert_eq!(a.device().type_id(), device.type_id());
        assert_eq!(a.read(), [2, 4, 6]);

        use custos::UnifiedMemChain;

        let a = custos::cl_cpu_exec_unified!(
            device, lhs, rhs; device.cpu.add(&lhs, &rhs)
        )?;

        assert_eq!(a.device().type_id(), device.type_id());
        assert_eq!(a.read(), [2, 4, 6]);

        Ok(())
    }
}
