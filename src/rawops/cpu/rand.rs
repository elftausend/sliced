use custos::{prelude::Float, Buffer, MainMemory, Shape, CPU};

use crate::RandOp;

pub fn rand_slice<T: PartialOrd + Copy + Float>(slice: &mut [T], lo: T, hi: T) {
    let rng = fastrand::Rng::new();
    for value in slice {
        *value = T::as_generic(rng.f64()) * (hi - (lo)) + (lo);
    }
}

//#[impl_stack]
impl<T: Float, D: MainMemory, S: Shape> RandOp<T, S, D> for CPU {
    #[inline]
    fn rand(&self, x: &mut Buffer<T, D, S>, lo: T, hi: T) {
        rand_slice(x, lo, hi)
    }
}
