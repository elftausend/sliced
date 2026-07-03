use std::ops::{Deref, DerefMut};

use custos::{impl_stack, prelude::Float, Buffer, Device, Shape, Stack, WrappedData, CPU};

use crate::RandOp;

pub fn rand_slice<T: PartialOrd + Copy + Float>(slice: &mut [T], lo: T, hi: T) {
    let rng = fastrand::Rng::new();
    for value in slice {
        *value = T::as_generic(rng.f64()) * (hi - (lo)) + (lo);
    }
}

#[impl_stack]
impl<Mods: WrappedData, T, D, S> RandOp<T, S, D> for CPU<Mods>
where
    T: Float,
    S: Shape,
    D: Device,
    D::Base<T, S>: Deref<Target = [T]> + DerefMut,
{
    #[inline]
    fn rand(&self, x: &mut Buffer<T, D, S>, lo: T, hi: T) {
        rand_slice(x, lo, hi)
    }
}
