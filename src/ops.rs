use std::{
    fmt::Display,
    ops::{Add, Mul, Neg, Sub},
};

use custos::{
    prelude::{Float, One, Two},
    Alloc, ApplyFunction, Buffer, Combiner, Device, Eval, MayTapeReturn, Shape, UnaryGrad,
    WriteBuf, MayToCLSource,
};

use crate::{
    AddElementWiseGrad, BinaryElementWise, BinaryElementWiseGrad, Diagflat, DiagflatGrad, Gemm,
    GemmGrad, MaxCols, MaxColsGrad, MaxRows, MaxRowsGrad, MeanCols, MeanColsGrad, MeanRows,
    MeanRowsGrad, RowOp, RowOpGrad, Softmax, SoftmaxGrad, SumCols, SumColsGrad, SumRows,
    SumRowsGrad, TranposeGrad, Transpose,
};

pub trait SquareMayGrad<T, S = ()>: Device
where
    T: 'static,
    S: Shape,
{
    fn square(&self, x: &Buffer<T, Self, S>) -> Buffer<T, Self, S>
    where
        T: MayToCLSource + Eval<T> + Mul<Output = T> + Copy + Two,
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
                let (lhs, lhs_grad, out_grad) = grads.get_double(device, ids);

                device.add_unary_grad(&lhs, lhs_grad, out_grad, |x| x.mul(T::two()));
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
                let (lhs, lhs_grad, out_grad) = grads.get_double(device, ids);

                device.add_unary_grad(&lhs, lhs_grad, out_grad, |x| x.pow(rhs - T::one()).mul(rhs))
            });
        }

        out
    }
}

impl<T: 'static, S: Shape, D: Device> SquareMayGrad<T, S> for D {}

pub trait BinaryOpsMayGrad<T, S: Shape = (), D: Device = Self>: Device {
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>;
    fn add2(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>
    where
        D: AddElementWiseGrad<T>;
    fn sub(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>;
    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>;
}

impl<T, S, D> BinaryOpsMayGrad<T, S, D> for D
where
    S: Shape + 'static,
    D: BinaryElementWise<T, S, D>
        + BinaryElementWiseGrad<T, (), D>
        + MayTapeReturn
        + for<'b> Alloc<'b, T>,
    T: Mul<Output = T>
        + Sub<Output = T>
        + Add<Output = T>
        + Neg<Output = T>
        + MayToCLSource
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
                let (lhs, rhs, lhs_grad, rhs_grad, out_grad) =
                    grads.get_triple::<T, ()>(device, ids);

                device.binary_ew_grad(
                    &lhs,
                    &rhs,
                    lhs_grad,
                    rhs_grad,
                    out_grad,
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
                let (lhs, rhs, lhs_grad, rhs_grad, out_grad) =
                    grads.get_triple::<T, ()>(device, ids);

                device.binary_ew_grad(
                    &lhs,
                    &rhs,
                    lhs_grad,
                    rhs_grad,
                    out_grad,
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
                let (lhs, rhs, lhs_grad, rhs_grad, out_grad) =
                    grads.get_triple::<T, ()>(device, ids);

                device.binary_ew_grad(
                    &lhs,
                    &rhs,
                    lhs_grad,
                    rhs_grad,
                    out_grad,
                    |_, rhs| rhs,
                    |lhs, _| lhs,
                );
            });
        }

        out
    }

    fn add2(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>
    where
        D: AddElementWiseGrad<T>,
    {
        let out = self.binary_ew(lhs, rhs, |a, b| a.add(b));

        #[cfg(feature = "autograd")]
        {
            let ids = (lhs.id(), rhs.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (_, _, lhs_grad, rhs_grad, out_grad) = grads.get_triple::<T, ()>(device, ids);

                device.add_ew_grad(lhs_grad, rhs_grad, out_grad);
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
    T: Clone + 'static,
    IS: Shape,
    OS: Shape,
    D: Transpose<T, IS, OS> + TranposeGrad<T> + MayTapeReturn + for<'b> Alloc<'b, T> + WriteBuf<T>,
{
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.transpose(rows, cols, x);

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (_, lhs_grad, out_grad) = grads.get_double(device, ids);

                device.transpose_grad(cols, rows, lhs_grad, out_grad);
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
    T: 'static,
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
                let (lhs, rhs, lhs_grad, rhs_grad, out_grad) =
                    grads.get_triple::<T, ()>(device, ids);

                device.gemm_grad(m, k, n, &lhs, &rhs, lhs_grad, rhs_grad, out_grad);
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
    T: 'static + Add<Output = T>,
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
        let out = self.add_row(cols, lhs, rhs);

        #[cfg(feature = "autograd")]
        {
            let ids = (lhs.id(), rhs.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (_, _, lhs_grad, rhs_grad, out_grad) = grads.get_triple::<T, ()>(device, ids);

                device.add_row_grad(rows, cols, lhs_grad, rhs_grad, out_grad);
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
                let (_, rhs_grad, out_grad) = grads.get_double(device, ids);
                device.add_row_mut_grad(rows, cols, rhs_grad, out_grad);
            });
        }
    }
}

pub trait ExpMayGrad<T, S>
where
    Self: ApplyFunction<T, S> + UnaryGrad<T> + MayTapeReturn + for<'b> Alloc<'b, T>,
    T: Float + 'static,
    S: Shape,
{
    fn exp(&self, x: &Buffer<T, Self, S>) -> Buffer<T, Self, S> {
        let out = self.apply_fn(x, |x| x.exp());

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (lhs, lhs_grad, out_grad) = grads.get_double::<T, _, _>(device, ids);
                device.add_unary_grad(&lhs, lhs_grad, out_grad, |x| x.exp());
            });
        }

        out
    }
}

impl<T, S, D> ExpMayGrad<T, S> for D
where
    T: Float + 'static,
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
    T: 'static,
    IS: Shape,
    OS: Shape,
    D: MaxCols<T, IS, OS> + MayTapeReturn + for<'a> Alloc<'a, T> + MaxColsGrad<T>,
{
    fn max_cols(&self, rows: usize, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.max_cols(rows, cols, x);

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let out = unsafe { device.get_existing_buf::<T, ()>(ids.1) };
                let (x, x_grad, out_grad) = grads.get_double(device, ids);
                device.max_cols_grad(cols, &out, &x, x_grad, out_grad);
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
    T: 'static,
    IS: Shape,
    OS: Shape,
    D: MaxRows<T, IS, OS> + MayTapeReturn + for<'a> Alloc<'a, T> + MaxRowsGrad<T>,
{
    fn max_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.max_rows(cols, x);

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let out = unsafe { device.get_existing_buf::<T, ()>(ids.1) };
                let (x, x_grad, out_grad) = grads.get_double(device, ids);
                device.max_rows_grad(cols, &out, &x, x_grad, out_grad);
            })
        }

        out
    }
}

pub trait SumRowsMayGrad<T, IS, OS>: Device
where
    T: 'static,
    IS: Shape,
    OS: Shape,
{
    fn sum_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

impl<T, IS, OS, D> SumRowsMayGrad<T, IS, OS> for D
where
    T: Copy + 'static,
    IS: Shape,
    OS: Shape,
    D: MayTapeReturn + SumRows<T, IS, OS> + SumRowsGrad<T> + for<'a> Alloc<'a, T>,
{
    fn sum_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.sum_rows(cols, x);

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (_, x_grad, out_grad) = grads.get_double(device, ids);

                device.sum_rows_grad(cols, x_grad, out_grad);
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
    T: Copy + 'static,
    IS: Shape,
    OS: Shape,
    D: MayTapeReturn + SumCols<T, IS, OS> + SumColsGrad<T> + for<'a> Alloc<'a, T>,
{
    fn sum_cols(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.sum_cols(cols, x);

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (_, x_grad, out_grad) = grads.get_double(device, ids);

                device.sum_cols_grad(cols, x_grad, out_grad);
            })
        }
        out
    }
}

pub trait MeanColsMayGrad<T, IS, OS>: Device
where
    IS: Shape,
    OS: Shape,
{
    fn mean_cols(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

impl<T, IS, OS, D> MeanColsMayGrad<T, IS, OS> for D
where
    T: Copy + 'static,
    IS: Shape,
    OS: Shape,
    D: MayTapeReturn + MeanCols<T, IS, OS> + MeanColsGrad<T> + for<'a> Alloc<'a, T>,
{
    fn mean_cols(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.mean_cols(cols, x);

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (_, x_grad, out_grad) = grads.get_double(device, ids);
                device.mean_cols_grad(cols, x_grad, out_grad);
            })
        }
        out
    }
}

pub trait MeanRowsMayGrad<T, IS, OS>: Device
where
    IS: Shape,
    OS: Shape,
{
    fn mean_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

impl<T, IS, OS, D> MeanRowsMayGrad<T, IS, OS> for D
where
    T: Copy + 'static,
    IS: Shape,
    OS: Shape,
    D: MayTapeReturn + MeanRows<T, IS, OS> + MeanRowsGrad<T> + for<'a> Alloc<'a, T>,
{
    fn mean_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.mean_rows(cols, x);

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (_, x_grad, out_grad) = grads.get_double(device, ids);
                device.mean_rows_grad(cols, x_grad, out_grad);
            })
        }
        out
    }
}

pub trait DiagflatMayGrad<T, IS, OS>: Device
where
    IS: Shape,
    OS: Shape,
{
    fn diagflat(&self, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

impl<T, IS: Shape, OS: Shape, D> DiagflatMayGrad<T, IS, OS> for D
where
    T: Copy + 'static,
    D: Diagflat<T, IS, OS>
        + DiagflatGrad<T, IS, OS>
        + for<'a> Alloc<'a, T, IS>
        + for<'a> Alloc<'a, T, OS>
        + MayTapeReturn,
{
    #[inline]
    fn diagflat(&self, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.diagflat(x);

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let (_, x_grad, out_grad) = grads.get_double(device, ids);
                device.diagflat_grad(x_grad, out_grad);
            })
        }
        out
    }
}

pub trait SoftmaxMayGrad<T, S>: Device
where
    S: Shape,
{
    fn softmax(
        &self,
        samples: usize,
        features: usize,
        x: &Buffer<T, Self, S>,
    ) -> Buffer<T, Self, S>;
}

impl<T, S, D> SoftmaxMayGrad<T, S> for D
where
    T: Copy + 'static,
    S: Shape,
    D: Softmax<T, S> + SoftmaxGrad<T, S> + for<'a> Alloc<'a, T, S> + MayTapeReturn,
{
    fn softmax(
        &self,
        samples: usize,
        features: usize,
        x: &Buffer<T, Self, S>,
    ) -> Buffer<T, Self, S> {
        let out = self.softmax(samples, features, x);

        #[cfg(feature = "autograd")]
        {
            let ids = (x.id(), out.id());
            self.tape_mut().add_grad_fn(move |grads, device| {
                let out = unsafe { device.get_existing_buf(ids.1) };
                let (_, x_grad, out_grad) = grads.get_double(device, ids);
                device.softmax_grad(samples, features, x_grad, &out, out_grad);
            })
        }
        out
    }
}

macro_rules! _impl_may_autograd_op {
    ($trait_name:ident, $forward_trait:ident, $backward_trait:ident) => {};
}
