mod impl_from;
mod impl_from_const;

#[cfg(feature = "static-api")]
mod to_static_device;

use std::{fmt::Display, ops::Mul};

use custos::{
    prelude::{Float, Number, Numeric, Two},
    Alloc, ApplyFunction, Buffer, CloneBuf, Combiner, Device, IsShapeIndep, MayTapeActions, Shape,
    UnaryElementWiseMayGrad, UnaryGrad, CPU, OnNewBuffer,
};

use crate::{
    AddElementWiseGrad, BinaryElementWise, BinaryOpsMayGrad, DiagflatMayGrad, GemmMayGrad,
    MaxColsMayGrad, MaxRowsMayGrad, PowMayGrad, RandOp, RowOpMayGrad, SoftmaxMayGrad,
    SquareMayGrad, SumColsMayGrad, TransposeMayGrad,
};

pub struct Matrix<'a, T = f32, D: Device = CPU, S: Shape = ()> {
    data: Buffer<'a, T, D, S>,
    rows: usize,
    cols: usize,
}

impl<'a, T, D: Device + OnNewBuffer<T, D, S>, S: Shape> Matrix<'a, T, D, S> {
    #[inline]
    pub fn new(device: &'a D, rows: usize, cols: usize) -> Matrix<'a, T, D, S>
    where
        D: Alloc<T>,
    {
        Matrix {
            data: Buffer::new(device, rows * cols),
            rows,
            cols,
        }
    }

    /// Returns the rows of `Matrix`.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the columns of `Matrix`.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns a reference to the underlying buffer.
    #[inline]
    pub fn as_buf(&self) -> &Buffer<'a, T, D, S> {
        &self.data
    }

    #[inline]
    pub fn to_buf(self) -> Buffer<'a, T, D, S> {
        self.data
    }

    /// Returns a mutable reference to the underlying buffer.
    #[inline]
    pub fn as_buf_mut(&mut self) -> &mut Buffer<'a, T, D, S> {
        &mut self.data
    }

    #[allow(non_snake_case)]
    #[inline]
    pub fn T<OS: Shape>(&self) -> Matrix<'a, T, D, OS>
    where
        D: TransposeMayGrad<T, S, OS>,
    {
        Matrix {
            data: self.device().transpose(self.rows, self.cols, self),
            rows: self.cols,
            cols: self.rows,
        }
    }

    #[inline]
    pub fn gemm<RS: Shape, OS: Shape>(&self, rhs: &Matrix<'a, T, D, RS>) -> Matrix<'a, T, D, OS>
    where
        D: GemmMayGrad<T, S, RS, OS>,
    {
        (
            self.device()
                .gemm(self.rows, self.cols, rhs.cols, self, rhs),
            self.rows,
            rhs.cols,
        )
            .into()
    }

    #[inline]
    // TODO Add trait
    pub fn add(&self, rhs: &Matrix<'a, T, D, S>) -> Matrix<'a, T, D, S>
    where
        D: BinaryOpsMayGrad<T, S>,
    {
        (self.device().add(self, rhs), self.rows, self.cols).into()
    }

    #[inline]
    // TODO Mul trait
    pub fn mul(&self, rhs: &Matrix<'a, T, D, S>) -> Matrix<'a, T, D, S>
    where
        D: BinaryOpsMayGrad<T, S>,
    {
        (self.device().mul(self, rhs), self.rows, self.cols).into()
    }

    #[inline]
    pub fn add_row<RS: Shape>(&self, rhs: &Matrix<'a, T, D, RS>) -> Matrix<'a, T, D, S>
    where
        D: RowOpMayGrad<T, S, RS>,
    {
        (
            self.device().add_row(self.rows, self.cols, self, rhs),
            self.rows,
            self.cols,
        )
            .into()
    }

    #[inline]
    pub fn add_row_mut<RS: Shape>(&mut self, rhs: &Matrix<'a, T, D, RS>)
    where
        D: RowOpMayGrad<T, S, RS>,
    {
        self.device().add_row_mut(self.rows, self.cols, self, rhs)
    }

    #[inline]
    pub fn relu(&self) -> Matrix<'a, T, D, S>
    where
        T: Number + 'static,
        D: UnaryElementWiseMayGrad<T, D, S>
            + ApplyFunction<T, S>
            + MayTapeActions
            + UnaryGrad<T, S>
            + Alloc<T>
            + 'static,
    {
        let out = self.device().apply_fn(self, |x| x.geq(T::zero()).mul(x));

        #[cfg(feature = "autograd")]
        {
            let ids = (self.id(), out.id());
            self.device().tape_mut().add_grad_fn(move |grads, device| {
                let (lhs, mut lhs_grad, out_grad) = grads.get_double(device, ids);
                device.add_unary_grad(&lhs, &mut lhs_grad, &out_grad, |x| x.geq(T::zero()));
            });
        }

        (out, self.rows, self.cols).into()

        /*// TODO may inline
        // -> huge performance difference?
        // -> look at profiler again
        (
            self.device()
                .unary_ew(self, |x| x.geq(T::zero()).mul(x), |x| x.geq(T::zero())),
            self.rows,
            self.cols,
        )
            .into()*/
    }

    #[inline]
    pub fn rand(&mut self, lo: T, hi: T)
    where
        D: RandOp<T, S>,
    {
        self.device().rand(self, lo, hi);
    }

    #[inline]
    pub fn squared(&self) -> Matrix<'a, T, D, S>
    where
        T: Numeric + Mul<Output = T> + Copy + Two + Combiner + 'static,
        D: SquareMayGrad<T, S> + ApplyFunction<T, S> + UnaryGrad<T, S> + Alloc<T> + MayTapeActions,
    {
        (self.device().square(self), self.rows, self.cols).into()
    }

    #[inline]
    pub fn pow(&self, rhs: T) -> Matrix<'a, T, D, S>
    where
        D: PowMayGrad<T, S>,
    {
        (self.device().pow(self, rhs), self.rows, self.cols).into()
    }

    #[inline]
    pub fn max_rows<OS: Shape>(&self) -> Matrix<'a, T, D, OS>
    where
        D: MaxRowsMayGrad<T, S, OS>,
    {
        (
            self.device().max_rows(self.cols, self),
            self.rows,
            self.cols,
        )
            .into()
    }

    #[inline]
    pub fn max_cols<OS: Shape>(&self) -> Matrix<'a, T, D, OS>
    where
        D: MaxColsMayGrad<T, S, OS>,
    {
        (
            self.device().max_cols(self.rows, self.cols, self),
            self.rows,
            self.cols,
        )
            .into()
    }

    #[inline]
    pub fn sum_cols<OS: Shape>(&self) -> Matrix<'a, T, D, OS>
    where
        D: SumColsMayGrad<T, S, OS>,
    {
        (self.device().sum_cols(self.cols, self), self.rows, 1).into()
    }

    #[inline]
    pub fn l2_norm_cols<OS>(&self) -> Matrix<'a, T, D, OS>
    where
        T: Combiner + Float + 'static,
        OS: Shape + 'static,
        D: ApplyFunction<T, OS>
            + UnaryGrad<T, OS>
            + ApplyFunction<T, S>
            + UnaryGrad<T, S>
            + Alloc<T>
            + MayTapeActions
            + SumColsMayGrad<T, S, OS>,
    {
        self.squared().sum_cols().pow(T::one() / T::two())
    }

    #[inline]
    pub fn diagflat<OS: Shape>(&self) -> Matrix<'a, T, D, OS>
    where
        D: DiagflatMayGrad<T, S, OS>,
    {
        (self.device().diagflat(self), self.rows, self.rows).into()
    }

    #[inline]
    pub fn softmax(&self) -> Matrix<'a, T, D, S>
    where
        D: SoftmaxMayGrad<T, S>,
    {
        (
            self.device().softmax(self.rows, self.cols, self),
            self.rows,
            self.cols,
        )
            .into()
    }
}

impl<T, D: IsShapeIndep, S: Shape> Matrix<'_, T, D, S> {
    #[inline]
    pub fn as_dims<'b, O: Shape>(&self) -> &Matrix<'b, T, D, O> {
        unsafe { &*(self as *const Self).cast() }
    }

    #[inline]
    pub fn as_dims_mut<'b, O: Shape>(&mut self) -> &mut Matrix<'b, T, D, O> {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl<'a, T, D: Device, S: Shape> core::ops::Deref for Matrix<'a, T, D, S> {
    type Target = Buffer<'a, T, D, S>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &*self
    }
}

impl<'a, T, D: Device, S: Shape> core::ops::DerefMut for Matrix<'a, T, D, S> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self
    }
}

impl<'a, T, D: BinaryOpsMayGrad<T, S>, S: Shape> std::ops::Sub for &Matrix<'a, T, D, S> {
    type Output = Matrix<'a, T, D, S>;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        (self.device().sub(self, rhs), self.rows, self.cols).into()
    }
}

impl<'a, T, D: BinaryOpsMayGrad<T, S>, S: Shape> std::ops::Add for &Matrix<'a, T, D, S> {
    type Output = Matrix<'a, T, D, S>;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        (self.device().add(self, rhs), self.rows, self.cols).into()
    }
}

impl<'a, T, D: BinaryOpsMayGrad<T, S>, S: Shape> std::ops::Add for Matrix<'a, T, D, S> {
    type Output = Matrix<'a, T, D, S>;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        (self.device().add(&self, &rhs), self.rows, self.cols).into()
    }
}

impl<'a, T: Clone, D: CloneBuf<'a, T, S>, S: Shape> Clone for Matrix<'a, T, D, S> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            rows: self.rows.clone(),
            cols: self.cols.clone(),
        }
    }
}
