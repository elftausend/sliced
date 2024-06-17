use custos::{prelude::Float, AddOperation, Buffer, Retrieve, SetOpHint, CPU};

use crate::{ColOp, Exp, MaxCols, Softmax, SumCols};

impl<T, Mods> Softmax<T> for CPU<Mods>
where
    Mods: Retrieve<Self, T, ()> + AddOperation + SetOpHint<T> + 'static,
    T: Float + PartialOrd,
{
    #[inline]
    fn softmax(&self, samples: usize, sample_size: usize, x: &Buffer<T, Self>) -> Buffer<T, Self> {
        let max_cols = self.max_cols(samples, sample_size, x);
        let x = &self.sub_cols(sample_size, x, &max_cols);
        let exp = self.exp(x);
        self.div_cols(sample_size, &exp, &self.sum_cols(sample_size, &exp))
    }
}

#[cfg(test)]
mod tests {
    use crate::Softmax;
    use custos::{Buffer, CPU};

    #[test]
    fn test_softmax() {
        let device = CPU::<custos::Autograd<custos::Base>>::new();
        let x = Buffer::from((&device, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let out = device.softmax(2, 3, &x);
        crate::test_utils::roughly_equals(
            &*out,
            &[
                0.09003057, 0.24472847, 0.66524096, 0.09003057, 0.24472847, 0.66524096,
            ],
        );
    }
}
