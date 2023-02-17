use std::ops::Sub;

use custos::{prelude::Float, Buffer, Device, MainMemory, Shape, CPU};

use crate::{ColOp, Exp, MaxCols, Softmax};

impl<T, S: Shape, D: MainMemory> Softmax<T, S, D> for CPU {
    #[inline]
    fn softmax(&self, samples: usize, sample_size: usize, x: &Buffer<T, D, S>) -> Buffer<T, D, S> {
        //let exp = self.exp(&self.sub_col(inputs, &self.max_cols(inputs)));
        todo!()
    }
}

fn softmax<T>(device: &CPU, samples: usize, sample_size: usize, x: &Buffer<T, CPU>)
where
    T: Float + Ord,
{
    let x = device.exp(&device.sub_cols(sample_size, x, &device.max_cols(samples, sample_size, x)));
}
