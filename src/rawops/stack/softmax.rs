use crate::{Exp, Softmax};
use custos::{prelude::Float, Base, Buffer, Device, Dim2, Stack, CPU};

impl<T, const SAMPLES: usize, const FEATURES: usize> Softmax<T, Dim2<SAMPLES, FEATURES>> for Stack
where
    T: Float,
{
    fn softmax(
        &self,
        _samples: usize,
        _features: usize,
        x: &Buffer<T, Self, Dim2<SAMPLES, FEATURES>>,
    ) -> Buffer<'_, T, Self, Dim2<SAMPLES, FEATURES>> {
        let cpu = CPU::<Base>::new();
        let cpu_x = cpu.buffer(x.read().as_flattened());
        let cpu_out = cpu.softmax(SAMPLES, FEATURES, &cpu_x);
        let mut out = Buffer::new(self, SAMPLES * FEATURES);
        out.write(&cpu_out);
        out
    }
}
