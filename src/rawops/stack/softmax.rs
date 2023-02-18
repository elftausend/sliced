use crate::{Exp, Softmax};
use custos::{prelude::Float, Buffer, Dim2, Stack};

impl<T, const SAMPLES: usize, const FEATURES: usize> Softmax<T, Dim2<SAMPLES, FEATURES>> for Stack
where
    T: Float,
{
    fn softmax(
        &self,
        _samples: usize,
        _features: usize,
        x: &Buffer<T, Self, Dim2<SAMPLES, FEATURES>>,
    ) -> Buffer<T, Self, Dim2<SAMPLES, FEATURES>> {
        Stack.exp(x);
        //Stack.max_cols()
        todo!()
    }
}
