use std::ops::AddAssign;

use crate::{diagflat, diagflat_grad, Diagflat, DiagflatGrad};
use custos::{Buffer, Device, Dim1, Dim2, Retriever, Stack};

impl<T: Copy + Default, const N: usize> Diagflat<T, Dim1<N>, Dim2<N, N>> for Stack {
    fn diagflat(&self, x: &Buffer<T, Self, Dim1<N>>) -> Buffer<T, Self, Dim2<N, N>> {
        let mut out = self.retrieve(x.len() * x.len(), x).unwrap();
        diagflat(x, &mut out);
        out
    }
}

impl<T: Copy + Default, const N: usize> Diagflat<T, Dim2<N, 1>, Dim2<N, N>> for Stack {
    fn diagflat(&self, x: &Buffer<T, Self, Dim2<N, 1>>) -> Buffer<T, Self, Dim2<N, N>> {
        let mut out = self.retrieve(x.len() * x.len(), x).unwrap();
        diagflat(x, &mut out);
        out
    }
}

impl<T: AddAssign + Copy, const N: usize> DiagflatGrad<T, Dim1<N>, Dim2<N, N>> for Stack {
    #[inline]
    fn diagflat_grad(
        &self,
        x_grad: &mut Buffer<T, Self, Dim1<N>>,
        out_grad: &Buffer<T, Self, Dim2<N, N>>,
    ) {
        diagflat_grad(x_grad, out_grad);
    }
}

impl<T: AddAssign + Copy, const N: usize> DiagflatGrad<T, Dim2<N, 1>, Dim2<N, N>> for Stack {
    #[inline]
    fn diagflat_grad(
        &self,
        x_grad: &mut Buffer<T, Self, Dim2<N, 1>>,
        out_grad: &Buffer<T, Self, Dim2<N, N>>,
    ) {
        diagflat_grad(x_grad, out_grad);
    }
}
