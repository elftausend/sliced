use std::ops::Deref;

use custos::{impl_stack, Buffer, Device, Eval, MayToCLSource, Resolve, Shape, ToVal, CPU, Retriever};

use super::BinaryElementWise;

#[cfg(feature = "stack")]
use custos::Stack;

#[impl_stack]
impl<T, S, D> BinaryElementWise<T, S, D> for CPU
where
    T: Copy + Default,
    S: Shape,
    D: Device,
    D::Data<T, S>: Deref<Target = [T]>,
{
    #[inline]
    fn binary_ew<O>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        f: impl Fn(Resolve<T>, Resolve<T>) -> O,
    ) -> Buffer<T, Self, S>
    where
        O: Eval<T> + MayToCLSource,
    {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));
        slice_binary_ew(lhs, rhs, &mut out, f);
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
