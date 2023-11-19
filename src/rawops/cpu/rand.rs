use std::ops::{Deref, DerefMut};

use custos::{prelude::Float, Buffer, Device, Shape, CPU, OnDropBuffer};

use crate::RandOp;

pub fn rand_slice<T: PartialOrd + Copy + Float>(slice: &mut [T], lo: T, hi: T) {
    let rng = fastrand::Rng::new();
    for value in slice {
        *value = T::as_generic(rng.f64()) * (hi - (lo)) + (lo);
    }
}

//#[impl_stack]
impl<Mods: OnDropBuffer, T, D, S> RandOp<T, S, D> for CPU<Mods>
where
    T: Float,
    S: Shape,
    D: Device,
    D::Data<T, S>: Deref<Target = [T]> + DerefMut,
{
    #[inline]
    fn rand(&self, x: &mut Buffer<T, D, S>, lo: T, hi: T) {
        rand_slice(x, lo, hi)
    }
}
