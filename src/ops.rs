use std::{
    fmt::Display,
    ops::{Add, Mul},
};

use custos::{
    prelude::{One, Two},
    Alloc, ApplyFunction, Buffer, Combiner, Device, Eval, MayTapeReturn, Shape, UnaryGrad,
};

use crate::{BinaryElementWise, BinaryGrad};

pub trait Square<T, S = ()>: Device
where
    S: Shape,
{
    fn square(&self, x: &Buffer<T, Self, S>) -> Buffer<T, Self, S>
    where
        T: Display + Eval<T> + Mul<Output = T> + Copy + Two,
        Self: ApplyFunction<T, S, Self>
            + UnaryGrad<T, S, Self>
            + MayTapeReturn
            + for<'b> Alloc<'b, T, S>,
    {
        let out = self.apply_fn(x, |x| x.mul(x));

        let ids = (x.id(), out.id());
        self.tape_mut().add_grad_fn(move |grads, device| {
            let (lhs, mut lhs_grad, out_grad) = grads.get_double::<T, S>(device, ids);

            device.add_unary_grad(&lhs, &mut lhs_grad, &out_grad, |x| x.mul(T::two()));
        });

        out
    }
}

impl<T, S: Shape, D: Device> Square<T, S> for D {}

pub trait BinaryOps<T, S: Shape = (), D: Device = Self>: Device {
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>;
    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>;
}

impl<T, S, D> BinaryOps<T, S, D> for D
where
    S: Shape + 'static,
    D: BinaryElementWise<T, S, D> + BinaryGrad<T, S, D> + MayTapeReturn + for<'b> Alloc<'b, T, S>,
    T: Mul<Output = T> + Add<Output = T> + Display + One + Eval<T> + 'static,
{
    #[inline]
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S> {
        let out = self.binary_ew(lhs, rhs, |a, b| a.add(b));

        let ids = (lhs.id(), rhs.id(), out.id());
        self.tape_mut().add_grad_fn(move |grads, device| {
            let (lhs, rhs, mut lhs_grad, mut rhs_grad, out_grad) =
                grads.get_triple::<T, S>(device, ids);

            device.add_binary_grad(
                &lhs,
                &rhs,
                &mut lhs_grad,
                &mut rhs_grad,
                &out_grad,
                |_, _| T::one(),
                |_, _| T::one(),
            );
        });

        out
    }

    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S> {
        let out = self.binary_ew(lhs, rhs, |a, b| a.mul(b));

        let ids = (lhs.id(), rhs.id(), out.id());
        self.tape_mut().add_grad_fn(move |grads, device| {
            let (lhs, rhs, mut lhs_grad, mut rhs_grad, out_grad) =
                grads.get_triple::<T, S>(device, ids);

            device.add_binary_grad(
                &lhs,
                &rhs,
                &mut lhs_grad,
                &mut rhs_grad,
                &out_grad,
                |_, rhs| rhs,
                |lhs, _| lhs,
            );
        });

        out
    }
}
