use custos::{prelude::Float, Buffer, CPU};

use crate::{ColOp, Exp, MaxCols, Softmax, SumCols};

impl<T> Softmax<T> for CPU
where
    T: Float + PartialOrd,
{
    #[inline]
    fn softmax(&self, samples: usize, sample_size: usize, x: &Buffer<T, Self>) -> Buffer<T, Self> {
        let exp = self.exp(&self.sub_cols(sample_size, x, &self.max_cols(samples, sample_size, x)));

        self.div_cols(sample_size, &exp, &self.sum_cols(sample_size, &exp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Softmax;
    use custos::{range, Buffer, Device, GenericBlas, Shape, CPU, MainMemory};

    #[test]
    fn test_softmax() {
        let device = CPU::new();
        let x = Buffer::from((&device, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let out = device.softmax(2, 3, &x);
        assert_eq!(&*out, [0.09003057, 0.24472847, 0.66524096, 0.09003057, 0.24472847, 0.66524096]);
    }
}
