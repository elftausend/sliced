use std::ops::{AddAssign, Mul};

use custos::{impl_stack, Buffer, Eval, MainMemory, Resolve, Shape, ToVal, CPU};

use super::BinaryGrad;

#[cfg(feature = "stack")]
use custos::Stack;

#[impl_stack]
impl<T, S, D> BinaryGrad<T, S, D> for CPU
where
    T: Copy + AddAssign + Mul<Output = T>,
    S: Shape,
    D: MainMemory,
{
    #[inline]
    fn add_binary_grad<LO, RO>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        lhs_grad: &mut Buffer<T, D, S>,
        rhs_grad: &mut Buffer<T, D, S>,
        out: &Buffer<T, D, S>,
        lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LO,
        rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RO,
    ) where
        LO: Eval<T> + ToString,
        RO: Eval<T> + ToString,
    {
        slice_binary_grad(lhs, rhs, lhs_grad, rhs_grad, out, lhs_grad_fn, rhs_grad_fn)
    }
}

pub fn slice_binary_grad<T, LO, RO>(
    lhs: &[T],
    rhs: &[T],
    lhs_grad: &mut [T],
    rhs_grad: &mut [T],
    out: &[T],
    lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LO,
    rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RO,
) where
    T: Copy + AddAssign + Mul<Output = T>,
    LO: Eval<T>,
    RO: Eval<T>,
{
    // TODO: measure performance
    let len = lhs.len().min(rhs.len()).min(out.len());

    for i in 0..len {
        lhs_grad[i] += lhs_grad_fn(lhs[i].to_val(), rhs[i].to_val()).eval() * out[i];
        rhs_grad[i] += rhs_grad_fn(lhs[i].to_val(), rhs[i].to_val()).eval() * out[i];
    }
}
