use std::ops::Deref;

use custos::{
    impl_stack, AddOperation, AsNoId, Buffer, Device, Eval, MayToCLSource, Resolve, Retrieve,
    Retriever, Shape, ToVal, CPU,
};

use super::BinaryElementWise;

#[cfg(feature = "stack")]
use custos::Stack;

#[impl_stack]
impl<T, S, D, Mods> BinaryElementWise<T, S, D> for CPU<Mods>
where
    T: Copy + Default + 'static,
    S: Shape,
    D: Device + AddOperation + 'static,
    D::Base<T, S>: Deref<Target = [T]>,
    Mods: Retrieve<Self, T, S> + AddOperation + 'static,
{
    #[inline]
    fn binary_ew<O>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        f: impl Fn(Resolve<T>, Resolve<T>) -> O + Copy + 'static,
    ) -> Buffer<T, Self, S>
    where
        O: Eval<T> + MayToCLSource,
    {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs)).unwrap();
        self.add_op((lhs, rhs, &mut out, f.no_id()), |(lhs, rhs, out, f)| {
            slice_binary_ew(lhs, rhs, out, **f);
            Ok(())
        })
        .unwrap();
        out
    }
}

pub fn slice_binary_ew<O, T>(
    lhs: &[T],
    rhs: &[T],
    out: &mut [T],
    f: impl Fn(Resolve<T>, Resolve<T>) -> O,
) where
    T: Copy,
    O: Eval<T> + MayToCLSource,
{
    for ((lhs, rhs), out) in lhs.iter().zip(rhs.iter()).zip(out.iter_mut()) {
        *out = f((*lhs).to_val(), (*rhs).to_val()).eval()
    }
}

#[cfg(test)]
mod tests {
    use custos::Combiner;

    use super::slice_binary_ew;

    #[test]
    fn test_slice_binary() {
        let lhs = &[6, 3, 7, 4, 3, 8, 10];
        let rhs = &[-3, 3, 8, 31, 4, 3, 2];

        let mut out = [0; 7];

        slice_binary_ew(lhs, rhs, &mut out, |a, b| a.add(b));

        assert_eq!(out, [3, 6, 15, 35, 7, 11, 12]);
    }
}
