use std::{
    fmt::Display,
    ops::{Add, Mul, Neg, Sub},
};

use custos::{
    prelude::{Float, One, Two},
    Alloc, ApplyFunction, Buffer, Combiner, Device, Eval, MayTapeReturn, Shape, UnaryGrad,
    WriteBuf,
};

use crate::{
    BinaryElementWise, BinaryGrad, Gemm, GemmGrad, MaxCols, MaxColsGrad, MaxRows, MaxRowsGrad,
    RowOp, RowOpGrad, SumCols, SumColsGrad, SumRows, SumRowsGrad, Transpose,
};

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

pub trait ExpMayGrad<T, S>
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

impl<T, S, D> ExpMayGrad<T, S> for D
where
    T: Float,
    S: Shape,
    D: ApplyFunction<T, S> + UnaryGrad<T> + MayTapeReturn + for<'b> Alloc<'b, T>,
{
}

pub trait Exp<T: Float, S: Shape>: ApplyFunction<T, S> {
    #[inline]
    fn exp(&self, x: &Buffer<T, Self, S>) -> Buffer<T, Self, S> {
        self.apply_fn(x, |x| x.exp())
    }
}

impl<T, S, D> Exp<T, S> for D
where
    D: ApplyFunction<T, S>,
    T: Float,
    S: Shape,
{
}

pub trait MaxColsMayGrad<T, IS, OS>: Device
where
    IS: Shape,
    OS: Shape,
{
    fn max_cols(&self, rows: usize, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

impl<T, IS, OS, D> MaxColsMayGrad<T, IS, OS> for D
where
    IS: Shape,
    OS: Shape,
    D: MaxCols<T, IS, OS> + MayTapeReturn + for<'a> Alloc<'a, T> + MaxColsGrad<T>,
{
    fn max_cols(&self, rows: usize, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.max_cols(rows, cols, x);

        #[cfg(feature = "autograd")]
        {
            let (xid, oid) = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let x = device.get_existing_buf::<T, ()>(xid);
                let mut x_grad = grads.get_like_raw(device, xid);

                let out = device.get_existing_buf::<T, ()>(oid);
                let out_grad = grads.get_like_raw(device, oid);
                device.max_cols_grad(cols, &out, &x, &mut x_grad, &out_grad);
            })
        }

        out
    }
}

pub trait MaxRowsMayGrad<T, IS, OS>: Device
where
    IS: Shape,
    OS: Shape,
{
    fn max_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

impl<T, IS, OS, D> MaxRowsMayGrad<T, IS, OS> for D
where
    IS: Shape,
    OS: Shape,
    D: MaxRows<T, IS, OS> + MayTapeReturn + for<'a> Alloc<'a, T> + MaxRowsGrad<T>,
{
    fn max_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.max_rows(cols, x);

        #[cfg(feature = "autograd")]
        {
            let (xid, oid) = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let x = device.get_existing_buf::<T, ()>(xid);
                let mut x_grad = grads.get_like_raw(device, xid);

                let out = device.get_existing_buf::<T, ()>(oid);
                let out_grad = grads.get_like_raw(device, oid);
                device.max_rows_grad(cols, &out, &x, &mut x_grad, &out_grad);
            })
        }

        out
    }
}

pub trait SumRowsMayGrad<T, IS, OS>: Device
where
    IS: Shape,
    OS: Shape,
{
    fn sum_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

impl<T, IS, OS, D> SumRowsMayGrad<T, IS, OS> for D
where
    T: Copy,
    IS: Shape,
    OS: Shape,
    D: MayTapeReturn + SumRows<T, IS, OS> + SumRowsGrad<T> + for<'a> Alloc<'a, T>,
{
    fn sum_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.sum_rows(cols, x);

        #[cfg(feature = "autograd")]
        {
            let (xid, oid) = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let mut x_grad = grads.get_like_raw(device, xid);
                let out_grad = grads.get_like_raw(device, oid);

                device.sum_rows_grad(cols, &mut x_grad, &out_grad);
            })
        }
        out
    }
}

pub trait SumColsMayGrad<T, IS, OS>: Device
where
    IS: Shape,
    OS: Shape,
{
    fn sum_cols(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

impl<T, IS, OS, D> SumColsMayGrad<T, IS, OS> for D
where
    T: Copy,
    IS: Shape,
    OS: Shape,
    D: MayTapeReturn + SumCols<T, IS, OS> + SumColsGrad<T> + for<'a> Alloc<'a, T>,
{
    fn sum_cols(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.sum_cols(cols, x);

        #[cfg(feature = "autograd")]
        {
            let (xid, oid) = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let mut x_grad = grads.get_like_raw(device, xid);
                let out_grad = grads.get_like_raw(device, oid);

                device.sum_cols_grad(cols, &mut x_grad, &out_grad);
            })
        }
        out
    }
}
