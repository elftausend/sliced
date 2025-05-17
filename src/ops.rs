use std::{
    fmt::Display,
    ops::{Add, Mul, Neg, Sub},
};

use custos::{
    number::Numeric, prelude::{Float, One, Two}, AddGradFn, AddOperation, Alloc, ApplyFunction, AsNoId, Buffer, Combiner, Device, Eval, HasId, MayGradActions, MayToCLSource, SetOpHint, Shape, TwoWay, UnaryGrad, Unit, WriteBuf, ZeroGrad
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
            + MayGradActions
            + Alloc<T>
            + ZeroGrad<T>
            + AddGradFn
            + 'static,
    {
        let out = self.apply_fn(x, |x| x.mul(x));

        self.add_grad_fn((x, &out), |(x, out)| unsafe {
            x.device()
                .add_unary_grad(x, x.grad_mut_unbound(), out.grad(), |x| x.mul(T::two()));
            Ok(())
        });

        out
    }
}

pub trait PowMayGrad<T: Unit, S: Shape = (), D: Device = Self>: Device {
    fn pow(&self, x: &Buffer<T, D, S>, rhs: T) -> Buffer<T, D, S>;
}

impl<T, S, D> PowMayGrad<T, S, D> for D
where
    T: TwoWay<T> + Display + Eval<T> + Mul<Output = T> + Float + 'static,
    S: Shape + 'static,
    D: ApplyFunction<T, S, Self>
        + UnaryGrad<T, S, Self>
        + MayGradActions
        + Alloc<T>
        + ZeroGrad<T>
        + AddGradFn
        + 'static,
{
    fn pow(&self, x: &Buffer<T, D, S>, rhs: T) -> Buffer<T, D, S> {
        let out = self.apply_fn(x, move |x| x.pow(rhs));

        self.add_grad_fn((x, &out), move |(x, out)| unsafe {
            x.device()
                .add_unary_grad(x, x.grad_mut_unbound(), out.grad(), move |x| {
                    x.pow(rhs - T::one()).mul(rhs)
                });
            Ok(())
        });

        out
    }
}

impl<T: 'static, S: Shape, D: Device> SquareMayGrad<T, S> for D {}

pub trait BinaryOpsMayGrad<T: Unit, S: Shape = (), D: Device = Self>: Device {
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>;

    fn add2(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>
    where
        D: AddElementWiseGrad<T, S>;

    fn sub(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>;

    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>;
}

impl<T, S, D> BinaryOpsMayGrad<T, S, D> for D
where
    S: Shape + 'static,
    D: BinaryElementWise<T, S, D>
        + BinaryElementWiseGrad<T, S, D>
        + MayGradActions
        + Alloc<T>
        + ZeroGrad<T>
        + AddGradFn
        + 'static,
    T: Mul<Output = T>
        + Sub<Output = T>
        + Add<Output = T>
        + Neg<Output = T>
        // + MayToCLSource
        + One
        // + Eval<T>
        + Numeric
        + 'static,
{
    #[inline]
    fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S> {
        let out = self.add(lhs, rhs);

        self.add_grad_fn((lhs, rhs, &out), |(lhs, rhs, out)| unsafe {
            lhs.device().binary_ew_grad(
                lhs,
                rhs,
                lhs.grad_mut_unbound(),
                rhs.grad_mut_unbound(),
                out.grad(),
                |_, _| T::one(),
                |_, _| T::one(),
            );
            Ok(())
        });

        out
    }

    fn sub(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S> {
        let out = self.sub(lhs, rhs);

        self.add_grad_fn((lhs, rhs, &out), |(lhs, rhs, out)| unsafe {
            lhs.device().binary_ew_grad(
                lhs,
                rhs,
                lhs.grad_mut_unbound(),
                rhs.grad_mut_unbound(),
                out.grad(),
                |_, _| T::one(),
                |_, _| -T::one(),
            );
            Ok(())
        });

        out
    }

    fn mul(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S> {
        let out = self.mul(lhs, rhs);

        self.add_grad_fn((lhs, rhs, &out), |(lhs, rhs, out)| unsafe {
            lhs.device().binary_ew_grad(
                lhs,
                rhs,
                lhs.grad_mut_unbound(),
                rhs.grad_mut_unbound(),
                out.grad(),
                |_, rhs| rhs,
                |lhs, _| lhs,
            );

            Ok(())
        });

        out
    }

    fn add2(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, D, S>
    where
        D: AddElementWiseGrad<T, S>,
    {
        let out = self.binary_ew(lhs, rhs, |a, b| a.add(b));

        self.add_grad_fn((lhs, rhs, &out), |(lhs, rhs, out)| unsafe {
            lhs.device()
                .add_ew_grad(lhs.grad_mut_unbound(), rhs.grad_mut_unbound(), out.grad());
            Ok(())
        });

        out
    }
}

pub trait TransposeMayGrad<T: Unit, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, D, OS>;
}

impl<T, IS, OS, D> TransposeMayGrad<T, IS, OS> for D
where
    T: Clone + 'static,
    IS: Shape,
    OS: Shape,
    D: Transpose<T, IS, OS>
        + TranposeGrad<T, IS, OS>
        + MayGradActions
        + Alloc<T>
        + ZeroGrad<T>
        + WriteBuf<T>
        + AddGradFn
        + 'static,
{
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.transpose(rows, cols, x);

        self.add_grad_fn(
            (x, &out),
            move |(x, out)| unsafe {
                x.device()
                    .transpose_grad(cols, rows, x.grad_mut_unbound(), out.grad());
                Ok(())
            },
        );

        out
    }
}

pub trait GemmMayGrad<T: Unit, LS: Shape = (), RS: Shape = (), OS: Shape = (), D: Device = Self>:
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
    D: Gemm<T, LS, RS, OS>
        + GemmGrad<T, LS, RS, OS>
        + MayGradActions
        + Alloc<T>
        + ZeroGrad<T>
        + AddGradFn
        + 'static,
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

        self.add_grad_fn(
            (lhs, rhs, &out),
            move |(lhs, rhs, out)| {
                lhs.device().gemm_grad(
                    m,
                    k,
                    n,
                    lhs,
                    rhs,
                    lhs.grad_mut(),
                    rhs.grad_mut(),
                    out.grad(),
                );
                Ok(())
            },
        );

        // #[cfg(feature = "autograd")]
        // {
        //     let ids = (lhs.id(), rhs.id(), out.id());
        //     self.tape_mut().add_grad_fn(move |grads, device| {
        //         let (lhs, rhs, lhs_grad, rhs_grad, out_grad) =
        //             grads.get_triple::<T, ()>(device, ids);

        //         device.gemm_grad(m, k, n, &lhs, &rhs, lhs_grad, rhs_grad, out_grad);
        //     });
        // }

        out
    }
}

pub trait RowOpMayGrad<T: Unit, LS: Shape = (), RS: Shape = (), D: Device = Self>: Device {
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
    D: RowOp<T, LS, RS>
        + RowOpGrad<T, LS, RS>
        + MayGradActions
        + Alloc<T>
        + AddGradFn
        + 'static
        + ZeroGrad<T>,
{
    fn add_row(
        &self,
        rows: usize,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, D, LS> {
        let out = self.add_row(cols, lhs, rhs);

        self.add_grad_fn(
            (rows.no_id(), cols.no_id(), lhs, rhs, &out),
            |(rows, cols, lhs, rhs, out)| {
                lhs.device().add_row_grad(
                    **rows,
                    **cols,
                    lhs.grad_mut(),
                    rhs.grad_mut(),
                    out.grad(),
                );
                Ok(())
            },
        );

        // #[cfg(feature = "autograd")]
        // {
        //     let ids = (lhs.id(), rhs.id(), out.id());
        //     self.tape_mut().add_grad_fn(move |grads, device| {
        //         let (_, _, lhs_grad, rhs_grad, out_grad) = grads.get_triple::<T, ()>(device, ids);

        //         device.add_row_grad(rows, cols, lhs_grad, rhs_grad, out_grad);
        //     });
        // }
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

        self.add_grad_fn(
            (rows.no_id(), cols.no_id(), lhs, rhs),
            |(rows, cols, lhs, rhs)| {
                lhs.device()
                    .add_row_mut_grad(**rows, **cols, rhs.grad_mut(), lhs.grad());
                Ok(())
            },
        );

        // #[cfg(feature = "autograd")]
        // {
        //     let ids = (rhs.id(), lhs.id());
        //     // FIXME may not work
        //     self.tape_mut().add_grad_fn(move |grads, device| {
        //         let (_, rhs_grad, out_grad) = grads.get_double(device, ids);
        //         device.add_row_mut_grad(rows, cols, rhs_grad, out_grad);
        //     });
        // }
    }
}

pub trait ExpMayGrad<T, S>
where
    Self: ApplyFunction<T, S> + UnaryGrad<T> + MayGradActions + Alloc<T>,
    T: Float + 'static,
    S: Shape,
{
    fn exp(&self, x: &Buffer<T, Self, S>) -> Buffer<T, Self, S> {
        let out = self.apply_fn(x, |x| x.exp());

        unimplemented!();
        // #[cfg(feature = "autograd")]
        // {
        //     let ids = (x.id(), out.id());
        //     self.tape_mut().add_grad_fn(move |grads, device| {
        //         let (lhs, lhs_grad, out_grad) = grads.get_double::<T, _, _>(device, ids);
        //         device.add_unary_grad(&lhs, lhs_grad, out_grad, |x| x.exp());
        //     });
        // }

        out
    }
}

impl<T, S, D> ExpMayGrad<T, S> for D
where
    T: Float + 'static,
    S: Shape,
    D: ApplyFunction<T, S> + UnaryGrad<T> + MayGradActions + Alloc<T>,
{
}

pub trait Clip<T: TwoWay<T> + Float, S: Shape>: ApplyFunction<T, S> {
    #[inline]

    fn clip(&self, x: &Buffer<T, Self, S>, min: T, max: T) -> Buffer<T, Self, S> {
        self.apply_fn(x, move |x| x.max(min).min(max))
    }
}

impl<T, S, D> Clip<T, S> for D
where
    D: ApplyFunction<T, S>,
    T: TwoWay<T> + Float,
    S: Shape,
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

pub trait MaxColsMayGrad<T: Unit, IS, OS>: Device
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
    D: ZeroGrad<T>
        + MaxCols<T, IS, OS>
        + MayGradActions
        + Alloc<T>
        + MaxColsGrad<T, IS, OS>
        + AddGradFn
        + 'static,
{
    fn max_cols(&self, rows: usize, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.max_cols(rows, cols, x);

        self.add_grad_fn((x, &mut out, cols.no_id()), |(x, out, cols)| {
            x.device()
                .max_cols_grad(**cols, out, x, x.grad_mut(), out.grad());
            Ok(())
        });

        // #[cfg(feature = "autograd")]
        // {
        //     let ids = (x.id(), out.id());
        //     self.tape_mut().add_grad_fn(move |grads, device| {
        //         let out = unsafe { device.get_existing_buf::<T, ()>(ids.1) };
        //         let (x, x_grad, out_grad) = grads.get_double(device, ids);
        //         device.max_cols_grad(cols, &out, &x, x_grad, out_grad);
        //     })
        // }

        out
    }
}

pub trait MaxRowsMayGrad<T: Unit, IS, OS>: Device
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
    D: MaxRows<T, IS, OS>
        + MayGradActions
        + Alloc<T>
        + MaxRowsGrad<T, IS, OS>
        + AddGradFn
        + ZeroGrad<T>
        + 'static,
{
    fn max_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.max_rows(cols, x);

        self.add_grad_fn((x, &mut out, cols.no_id()), |(x, out, cols)| {
            x.device()
                .max_rows_grad(**cols, out, x, x.grad_mut(), out.grad());
            Ok(())
        });
        // #[cfg(feature = "autograd")]
        // {
        //     let ids = (x.id(), out.id());
        //     self.tape_mut().add_grad_fn(move |grads, device| {
        //         let out = unsafe { device.get_existing_buf::<T, ()>(ids.1) };
        //         let (x, x_grad, out_grad) = grads.get_double(device, ids);
        //         device.max_rows_grad(cols, &out, &x, x_grad, out_grad);
        //     })
        // }

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
    D: MayGradActions + SumRows<T, IS, OS> + SumRowsGrad<T> + Alloc<T>,
{
    fn sum_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.sum_rows(cols, x);

        unimplemented!();
        // #[cfg(feature = "autograd")]
        // {
        //     let ids = (x.id(), out.id());
        //     self.tape_mut().add_grad_fn(move |grads, device| {
        //         let (_, x_grad, out_grad) = grads.get_double(device, ids);

        //         device.sum_rows_grad(cols, x_grad, out_grad);
        //     })
        // }
        out
    }
}

pub trait SumColsMayGrad<T: Unit, IS, OS>: Device
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
    D: MayGradActions
        + SumCols<T, IS, OS>
        + SumColsGrad<T, IS, OS>
        + Alloc<T>
        + ZeroGrad<T>
        + AddOperation
        + AddGradFn
        + 'static,
{
    fn sum_cols(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.sum_cols(cols, x);

        self.add_grad_fn((x, &out, cols.no_id()), |(x, out, cols)| {
            x.device().sum_cols_grad(**cols, x.grad_mut(), out.grad());
            Ok(())
        });

        // unimplemented!();
        // #[cfg(feature = "autograd")]
        // {
        //     let ids = (x.id(), out.id());
        //     self.tape_mut().add_grad_fn(move |grads, device| {
        //         let (_, x_grad, out_grad) = grads.get_double(device, ids);

        //         device.sum_cols_grad(cols, x_grad, out_grad);
        //     })
        // }
        out
    }
}

pub trait MeanColsMayGrad<T: Unit, IS, OS>: Device
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
    D: ZeroGrad<T>
        + MayGradActions
        + MeanCols<T, IS, OS>
        + MeanColsGrad<T, IS, OS>
        + Alloc<T>
        + AddGradFn
        + 'static,
{
    fn mean_cols(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.mean_cols(cols, x);

        self.add_grad_fn((x, &out, cols.no_id()), |(x, out, cols)| {
            x.device().mean_cols_grad(**cols, x.grad_mut(), out.grad());
            Ok(())
        });
        // #[cfg(feature = "autograd")]
        // {
        //     let ids = (x.id(), out.id());
        //     self.tape_mut().add_grad_fn(move |grads, device| {
        //         let (_, x_grad, out_grad) = grads.get_double(device, ids);
        //         device.mean_cols_grad(cols, x_grad, out_grad);
        //     })
        // }
        out
    }
}

pub trait MeanRowsMayGrad<T: Unit, IS, OS>: Device
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
    D: MayGradActions
        + MeanRows<T, IS, OS>
        + MeanRowsGrad<T, IS, OS>
        + Alloc<T>
        + ZeroGrad<T>
        + AddGradFn
        + 'static,
{
    fn mean_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.mean_rows(cols, x);

        self.add_grad_fn((cols.no_id(), x, &out), |(cols, x, out)| {
            x.device().mean_rows_grad(**cols, x.grad_mut(), out.grad());
            Ok(())
        });

        out
    }
}

pub trait DiagflatMayGrad<T: Unit, IS, OS>: Device
where
    IS: Shape,
    OS: Shape,
{
    fn diagflat(&self, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

impl<T, IS: Shape, OS: Shape, D> DiagflatMayGrad<T, IS, OS> for D
where
    T: Copy + 'static,
    D: AddGradFn
        + Diagflat<T, IS, OS>
        + DiagflatGrad<T, IS, OS>
        + Alloc<T>
        + ZeroGrad<T>
        + MayGradActions
        + 'static,
{
    #[inline]
    fn diagflat(&self, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let out = self.diagflat(x);

        self.add_grad_fn((x, &out), |(x, out)| {
            x.device().diagflat_grad(x.grad_mut(), out.grad());
            Ok(())
        });

        // #[cfg(feature = "autograd")]
        // {
        //     let ids = (x.id(), out.id());
        //     self.tape_mut().add_grad_fn(move |grads, device| {
        //         let (_, x_grad, out_grad) = grads.get_double(device, ids);
        //         device.diagflat_grad(x_grad, out_grad);
        //     })
        // }
        out
    }
}

pub trait SoftmaxMayGrad<T: Unit, S>: Device
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
    D: Softmax<T, S>
        + SoftmaxGrad<T, S>
        + AddGradFn
        + Alloc<T>
        + MayGradActions
        + 'static
        + ZeroGrad<T>,
{
    fn softmax(
        &self,
        samples: usize,
        features: usize,
        x: &Buffer<T, Self, S>,
    ) -> Buffer<T, Self, S> {
        let out = self.softmax(samples, features, x);

        self.add_grad_fn(
            (samples.no_id(), features.no_id(), x, &out),
            |(samples, features, x, out)| {
                x.device()
                    .softmax_grad(**samples, **features, x.grad_mut(), out, out.grad());
                Ok(())
            },
        );

        // #[cfg(feature = "autograd")]
        // {
        //     let ids = (x.id(), out.id());
        //     self.tape_mut().add_grad_fn(move |grads, device| {
        //         let out = unsafe { device.get_existing_buf(ids.1) };
        //         let (_, x_grad, out_grad) = grads.get_double(device, ids);
        //         device.softmax_grad(samples, features, x_grad, &out, out_grad);
        //     })
        // }
        out
    }
}

macro_rules! _impl_may_autograd_op {
    ($trait_name:ident, $forward_trait:ident, $backward_trait:ident) => {};
}
