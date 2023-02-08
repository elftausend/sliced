use std::{fmt::Display, ops::Mul};

use custos::{
    prelude::{Number, Two},
    Alloc, ApplyFunction, Buffer, Combiner, Device, IsShapeIndep, MayTapeReturn, Shape,
    UnaryElementWiseMayGrad, UnaryGrad, CPU,
};

use crate::{BinaryOpsMayGrad, GemmMayGrad, RandOp, RowOpMayGrad, SquareMayGrad, TransposeMayGrad, PowMayGrad};

pub struct Matrix<'a, T = f32, D: Device = CPU, S: Shape = ()> {
    data: Buffer<'a, T, D, S>,
    rows: usize,
    cols: usize,
}

impl<'a, T, D: Device, S: Shape> Matrix<'a, T, D, S> {
    #[inline]
    pub fn new(device: &'a D, rows: usize, cols: usize) -> Matrix<'a, T, D, S>
    where
        D: Alloc<'a, T, S>,
    {
        Matrix {
            data: Buffer::new(device, rows * cols),
            rows,
            cols,
        }
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
        (
            self.device().transpose(self.rows, self.cols, self),
            self.cols,
            self.rows,
        )
            .into()
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
            + MayTapeReturn
            + UnaryGrad<T, S>
            + for<'b> Alloc<'b, T, S>,
    {
        let out = self.device().apply_fn(self, |x| x.geq(T::zero()).mul(x));

        #[cfg(feature = "autograd")]
        {
            let ids = (self.id(), out.id());
            self.device().tape_mut().add_grad_fn(move |grads, device| {
                let (lhs, mut lhs_grad, out_grad) = grads.get_double::<T, S>(device, ids);
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
        T: Display + Mul<Output = T> + Copy + Two,
        D: SquareMayGrad<T, S>
            + ApplyFunction<T, S>
            + UnaryGrad<T, S>
            + for<'b> Alloc<'b, T, S>
            + MayTapeReturn
    {
        (self.device().square(self), self.rows, self.cols).into()
    }

    #[inline]
    pub fn pow(&self, rhs: T) -> Matrix<'a, T, D, S> 
    where
        D: PowMayGrad<T, S>
    {
        (self.device().pow(self, rhs), self.rows, self.cols).into()
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
        self.as_buf()
    }
}

impl<'a, T, D: Device, S: Shape> core::ops::DerefMut for Matrix<'a, T, D, S> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_buf_mut()
    }
}

impl<'a, T, D: Device, S: Shape> From<(Buffer<'a, T, D, S>, usize, usize)> for Matrix<'a, T, D, S> {
    #[inline]
    fn from((data, rows, cols): (Buffer<'a, T, D, S>, usize, usize)) -> Self {
        Matrix { data, rows, cols }
    }
}

impl<'a, T: Copy, D: Alloc<'a, T>, const N: usize> From<(&'a D, usize, usize, [T; N])>
    for Matrix<'a, T, D>
{
    #[inline]
    fn from((device, rows, cols, slice): (&'a D, usize, usize, [T; N])) -> Self {
        let data = Buffer::from((device, slice));
        Matrix { data, rows, cols }
    }
}

impl<'a, T: Copy, D: Alloc<'a, T>, const N: usize> From<(&'a D, usize, usize, &[T; N])>
    for Matrix<'a, T, D>
{
    #[inline]
    fn from((device, rows, cols, slice): (&'a D, usize, usize, &[T; N])) -> Self {
        let data = Buffer::from((device, slice));
        Matrix { data, rows, cols }
    }
}

// no tuple for dims
#[cfg(not(feature = "no-std"))]
// FIXME: In this case, GraphReturn acts as an "IsDynamic" trait, as GraphReturn is not implemented for Stack
// not anymore - but the message stays the same
impl<'a, T: Copy, D: Alloc<'a, T>> From<(&'a D, usize, usize, Vec<T>)> for Matrix<'a, T, D> {
    fn from((device, rows, cols, data): (&'a D, usize, usize, Vec<T>)) -> Self {
        let data = Buffer::from((device, data));
        Matrix { data, rows, cols }
    }
}

impl<'a, T, D: BinaryOpsMayGrad<T, S>, S: Shape> std::ops::Sub for &Matrix<'a, T, D, S> {
    type Output = Matrix<'a, T, D, S>;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        (self.device().sub(self, rhs), self.rows, self.cols).into()
    }
}
