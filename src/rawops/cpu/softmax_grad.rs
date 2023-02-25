use custos::{Shape, CPU, Buffer};
use crate::SoftmaxGrad;

impl<T, S: Shape> SoftmaxGrad<T, S> for CPU {
    fn softmax_grad(
        &self,
        samples: usize,
        feature: usize,
        x: &Buffer<T, Self, S>,
        x_grad: &Buffer<T, Self, S>,
        out_grad: &Buffer<T, Self, S>,
    ) {
        todo!()
    }
}