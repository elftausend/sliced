use custos::{Buffer, Device, Eval, MainMemory, Resolve, Shape, ToVal, CPU};

use crate::BinaryElementWise;

impl<T, S, D> BinaryElementWise<T, S, D> for CPU
where
    T: Copy,
    S: Shape,
    D: MainMemory,
{
    #[inline]
    fn binary_ew<O>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        f: impl Fn(Resolve<T>, Resolve<T>) -> O,
    ) -> Buffer<T, Self, S>
    where
        O: Eval<T> + ToString,
    {
        let mut out = self.retrieve(lhs.len());
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
    O: Eval<T> + ToString,
{
    for ((lhs, rhs), out) in lhs.iter().zip(rhs.iter()).zip(out.iter_mut()) {
        *out = f((*lhs).to_val(), (*rhs).to_val()).eval()
    }
}
