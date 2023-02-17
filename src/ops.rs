use std::{
    fmt::Display,
    ops::{Add, Mul, Neg, Sub},
};

use custos::{
    prelude::{Float, One, Two},
    Alloc, ApplyFunction, Buffer, Combiner, Device, Eval, MayTapeReturn, Shape, UnaryGrad,
    WriteBuf,
};

use crate::{BinaryElementWise, BinaryGrad, Gemm, GemmGrad, RowOp, RowOpGrad, Transpose};

pub trait SquareMayGrad<T, S = ()>: Device
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

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (lhs, mut lhs_grad, out_grad) = grads.get_double::<T, S>(device, ids);

                device.add_unary_grad(&lhs, &mut lhs_grad, &out_grad, |x| x.mul(T::two()));
            });
        }
        out
    }
}

pub trait PowMayGrad<T, S: Shape = (), D: Device = Self>: Device {
    fn pow(&self, x: &Buffer<T, D, S>, rhs: T) -> Buffer<T, D, S>;
}

impl<T, S, D> PowMayGrad<T, S, D> for D
where
    T: Display + Eval<T> + Mul<Output = T> + Float + 'static,
    S: Shape + 'static,
    D: ApplyFunction<T, S, Self> + UnaryGrad<T, S, Self> + MayTapeReturn + for<'b> Alloc<'b, T, S>,
{
    fn pow(&self, x: &Buffer<T, D, S>, rhs: T) -> Buffer<T, D, S> {
        let out = self.apply_fn(x, |x| x.pow(rhs));

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (lhs, mut lhs_grad, out_grad) = grads.get_double::<T, S>(device, ids);

                device.add_unary_grad(&lhs, &mut lhs_grad, &out_grad, |x| {
                    x.pow(rhs - T::one()).mul(rhs)
                })
            });
        }

        out
    }
}

impl<T, S: Shape, D: Device> SquareMayGrad<T, S> for D {}

pub trait BinaryOpsMayGrad<T, S: Shape = (), D: Device = Self>: Device {
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>;
    fn sub(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>;
    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>;
}

impl<T, S, D> BinaryOpsMayGrad<T, S, D> for D
where
    S: Shape + 'static,
    D: BinaryElementWise<T, S, D> + BinaryGrad<T, (), D> + MayTapeReturn + for<'b> Alloc<'b, T>,
    T: Mul<Output = T>
        + Sub<Output = T>
        + Add<Output = T>
        + Neg<Output = T>
        + Display
        + One
        + Eval<T>
        + 'static,
{
    #[inline]
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S> {
        let out = self.binary_ew(lhs, rhs, |a, b| a.add(b));

        #[cfg(feature = "autograd")]
        {
            let ids = (lhs.id(), rhs.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (lhs, rhs, mut lhs_grad, mut rhs_grad, out_grad) =
                    grads.get_triple::<T, ()>(device, ids);

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
        }
        out
    }

    fn sub(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S> {
        let out = self.binary_ew(lhs, rhs, |a, b| a.sub(b));

        #[cfg(feature = "autograd")]
        {
            let ids = (lhs.id(), rhs.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (lhs, rhs, mut lhs_grad, mut rhs_grad, out_grad) =
                    grads.get_triple::<T, ()>(device, ids);

                device.add_binary_grad(
                    &lhs,
                    &rhs,
                    &mut lhs_grad,
                    &mut rhs_grad,
                    &out_grad,
                    |_, _| T::one(),
                    |_, _| -T::one(),
                );
            });
        }

        out
    }

    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S> {
        let out = self.binary_ew(lhs, rhs, |a, b| a.mul(b));

        #[cfg(feature = "autograd")]
        {
            let ids = (lhs.id(), rhs.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (lhs, rhs, mut lhs_grad, mut rhs_grad, out_grad) =
                    grads.get_triple::<T, ()>(device, ids);

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
        }

        out
    }
}

pub trait TransposeMayGrad<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, D, OS>;
}

impl<T, IS, OS, D> TransposeMayGrad<T, IS, OS> for D
where
    T: Clone,
    IS: Shape,
    OS: Shape,
    D: Transpose<T, IS, OS> + MayTapeReturn + for<'b> Alloc<'b, T> + WriteBuf<T>,
{
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.transpose(rows, cols, x);

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (_, mut lhs_grad, out_grad) = grads.get_double::<T, ()>(device, ids);

                lhs_grad.write_buf(&out_grad);
            });
        }
        out
    }
}

pub trait GemmMayGrad<T, LS: Shape = (), RS: Shape = (), OS: Shape = (), D: Device = Self>:
    Device
{
    fn gemm(
        &self,
        m: usize,
        k: usize,
        n: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, D, OS>;
}

impl<T, LS, RS, OS, D> GemmMayGrad<T, LS, RS, OS> for D
where
    LS: Shape,
    RS: Shape,
    OS: Shape,
    D: Gemm<T, LS, RS, OS> + GemmGrad<T> + MayTapeReturn + for<'b> Alloc<'b, T>,
{
    fn gemm(
        &self,
        m: usize,
        k: usize,
        n: usize,
        lhs: &Buffer<T, Self, LS>,
        rhs: &Buffer<T, Self, RS>,
    ) -> Buffer<T, Self, OS> {
        let out = self.gemm(m, k, n, lhs, rhs);

        #[cfg(feature = "autograd")]
        {
            let ids = (lhs.id(), rhs.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (lhs, rhs, mut lhs_grad, mut rhs_grad, out_grad) =
                    grads.get_triple::<T, ()>(device, ids);

                device.gemm_grad(m, k, n, &lhs, &rhs, &mut lhs_grad, &mut rhs_grad, &out_grad);
            });
        }

        out
    }
}

pub trait RowOpMayGrad<T, LS: Shape = (), RS: Shape = (), D: Device = Self>: Device {
    fn add_row(
        &self,
        rows: usize,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, D, LS>;

    fn add_row_mut(
        &self,
        rows: usize,
        cols: usize,
        lhs: &mut Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    );
}

impl<T, LS, RS, D> RowOpMayGrad<T, LS, RS, D> for D
where
    LS: Shape,
    RS: Shape,
    D: RowOp<T, LS, RS> + RowOpGrad<T> + MayTapeReturn + for<'b> Alloc<'b, T>,
{
    fn add_row(
        &self,
        rows: usize,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, D, LS> {
        let out = self.add_row(rows, cols, lhs, rhs);

        #[cfg(feature = "autograd")]
        {
            let ids = (lhs.id(), rhs.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (_, _, mut lhs_grad, mut rhs_grad, out_grad) =
                    grads.get_triple::<T, ()>(device, ids);

                device.add_row_grad(rows, cols, &mut lhs_grad, &mut rhs_grad, &out_grad);
            });
        }
        out
    }

    fn add_row_mut(
        &self,
        rows: usize,
        cols: usize,
        lhs: &mut Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) {
        self.add_row_mut(rows, cols, lhs, rhs);

        #[cfg(feature = "autograd")]
        {
            let ids = (rhs.id(), lhs.id());
            // FIXME may not work
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (_, mut rhs_grad, out_grad) = grads.get_double(device, ids);
                device.add_row_mut_grad(rows, cols, &mut rhs_grad, &out_grad);
            });
        }
    }
}

pub trait Exp<T, S>
where
    Self: ApplyFunction<T, S> + UnaryGrad<T> + MayTapeReturn + for<'b> Alloc<'b, T>,
    T: Float,
    S: Shape,
{
    fn exp(&self, x: &Buffer<T, Self, S>) -> Buffer<T, Self, S> {
        let out = self.apply_fn(x, |x| x.exp());

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (lhs, mut lhs_grad, out_grad) = grads.get_double::<T, _>(device, ids);
                device.add_unary_grad(&lhs, &mut lhs_grad, &out_grad, |x| x.exp());
            });
        }

        out
    }
}

impl<T, S, D> Exp<T, S> for D
where
    T: Float,
    S: Shape,
    D: ApplyFunction<T, S> + UnaryGrad<T> + MayTapeReturn + for<'b> Alloc<'b, T>,
{
}
