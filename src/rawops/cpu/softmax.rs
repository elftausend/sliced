use custos::{Buffer, MainMemory, Shape, CPU};

use crate::Softmax;

impl<T, S: Shape, D: MainMemory> Softmax<T, S, D> for CPU {
    #[inline]
    fn softmax(&self, samples: usize, x: &Buffer<T, D, S>) -> Buffer<T, D, S> {
        todo!()
    }
}
