use custos::{prelude::Float, Buffer, CPU};

use crate::{ColOp, Exp, MaxCols, Softmax, SumCols};

impl<T> Softmax<T> for CPU
where
    T: Float + Ord,
{
    #[inline]
    fn softmax(&self, samples: usize, sample_size: usize, x: &Buffer<T, Self>) -> Buffer<T, Self> {
        let exp = self.exp(&self.sub_cols(sample_size, x, &self.max_cols(samples, sample_size, x)));

        self.div_cols(
            sample_size,
            &exp,
            &self.sum_cols(samples, sample_size, &exp),
        )
    }
}
