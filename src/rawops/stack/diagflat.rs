use std::ops::AddAssign;

use custos::{Stack, Dim1, Dim2, Buffer, Device};
use crate::{Diagflat, diagflat, DiagflatGrad, diagflat_grad};

impl<T: Copy + Default, const N: usize> Diagflat<T, Dim1<N>, Dim2<N, N>> for Stack {
    fn diagflat(&self, x: &Buffer<T, Self, Dim1<N>>) -> Buffer<T, Self, Dim2<N, N>> {
        let mut out = self.retrieve(x.len() * x.len());
        diagflat(x, &mut out);
        out
    }
}

impl<T: Copy + Default, const N: usize> Diagflat<T, Dim2<N, 1>, Dim2<N, N>> for Stack {
    fn diagflat(&self, x: &Buffer<T, Self, Dim2<N, 1>>) -> Buffer<T, Self, Dim2<N, N>> {
        let mut out = self.retrieve(x.len() * x.len());
        diagflat(x, &mut out);
        out
    }
}

impl<T: AddAssign + Copy, const N: usize> DiagflatGrad<T, Dim1<N>, Dim2<N, N>> for Stack {
    #[inline]
    fn diagflat_grad(&self, x_grad: &mut Buffer<T, Self, Dim1<N>>, out_grad: &Buffer<T, Self, Dim2<N, N>>) {
        diagflat_grad(x_grad, out_grad);
    }
}

impl<T: AddAssign + Copy, const N: usize> DiagflatGrad<T, Dim2<N, 1>, Dim2<N, N>> for Stack {
    #[inline]
    fn diagflat_grad(&self, x_grad: &mut Buffer<T, Self, Dim2<N, 1>>, out_grad: &Buffer<T, Self, Dim2<N, N>>) {
        diagflat_grad(x_grad, out_grad);
    }
}