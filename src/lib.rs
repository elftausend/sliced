mod devices;

use custos::{Buffer, Device, Eval, Resolve, Shape};

pub trait BinaryElementWise<T, S: Shape = (), D: Device = Self>: Device {
    fn binary_ew<O>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        f: impl Fn(Resolve<T>, Resolve<T>) -> O,
    ) -> Buffer<T, Self, S>
    where
        O: Eval<T> + ToString;
}
