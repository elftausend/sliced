use std::ops::{Div, Sub};

use custos::{Buffer, Device, Eval, MayDim2, Resolve, Shape};

mod cpu;

#[cfg(feature = "cpu")]
pub use cpu::*;

#[cfg(feature = "stack")]
mod stack;
#[cfg(feature = "stack")]
pub use stack::*;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(feature = "opencl")]
pub use opencl::*;

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

pub trait BinaryGrad<T, S: Shape = (), D: Device = Self>: Device {
    fn add_binary_grad<LO, RO>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        lhs_grad: &mut Buffer<T, D, S>,
        rhs_grad: &mut Buffer<T, D, S>,
        out: &Buffer<T, D, S>,
        lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LO,
        rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RO,
    ) where
        LO: Eval<T> + ToString,
        RO: Eval<T> + ToString;
}

pub trait BinaryEWMayGrad<T, S: Shape = (), D: Device = Self>: Device {
    fn binary_ew_w_grad<FO, LO, RO>(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
        forward_fn: impl Fn(Resolve<T>, Resolve<T>) -> FO,
        lhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> LO,
        rhs_grad_fn: impl Fn(Resolve<T>, Resolve<T>) -> RO,
    ) -> Buffer<T, Self, S>
    where
        FO: Eval<T> + ToString,
        LO: Eval<T> + ToString,
        RO: Eval<T> + ToString;
}

pub trait Gemm<T, LS: Shape = (), RS: Shape = (), OS: Shape = (), D: Device = Self>:
    Device
{
    fn gemm(
        &self,
        m: usize,
        k: usize,
        n: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, Self, OS>;
}

pub trait GemmGrad<T, LS: Shape = (), RS: Shape = (), OS: Shape = (), D: Device = Self>:
    Device
{
    fn gemm_grad(
        &self,
        m: usize,
        k: usize,
        n: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
        lhs_grad: &mut Buffer<T, D, LS>,
        rhs_grad: &mut Buffer<T, D, RS>,
        out_grad: &Buffer<T, D, OS>,
    );
}

pub trait Transpose<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub trait Transpose2<
    T,
    const ROWS: usize = 0,
    const COLS: usize = 0,
    IS: MayDim2<ROWS, COLS> = (),
    OS: MayDim2<COLS, ROWS> = (),
    D: Device = Self,
>: Device
{
    fn transpose(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub trait RowOp<T, LS: Shape = (), RS: Shape = (), D: Device = Self>: Device {
    fn add_row(
        &self,
        rows: usize,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, Self, LS>;
    fn add_row_mut(
        &self,
        rows: usize,
        cols: usize,
        lhs: &mut Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    );
}

pub trait ColOp<T, LS: Shape = (), RS: Shape = (), D: Device = Self>: Device {
    fn col_op<F>(
        &self,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
        f: F,
    ) -> Buffer<T, Self, LS>
    where
        F: Fn(T, T) -> T + Copy;

    #[inline]
    fn sub_cols(
        &self,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, Self, LS>
    where
        T: Sub<Output = T>,
    {
        self.col_op(cols, lhs, rhs, |lhs, rhs| lhs - rhs)
    }

    #[inline]
    fn div_cols(
        &self,
        cols: usize,
        lhs: &Buffer<T, D, LS>,
        rhs: &Buffer<T, D, RS>,
    ) -> Buffer<T, Self, LS>
    where
        T: Div<Output = T>,
    {
        self.col_op(cols, lhs, rhs, |lhs, rhs| lhs / rhs)
    }
}

pub trait RowOpGrad<T, LS: Shape = (), RS: Shape = (), D: Device = Self>: Device {
    fn add_row_grad(
        &self,
        rows: usize,
        cols: usize,
        lhs_grad: &mut Buffer<T, D, LS>,
        rhs_grad: &mut Buffer<T, D, RS>,
        out_grad: &Buffer<T, D, LS>,
    );
    fn add_row_mut_grad(
        &self,
        rows: usize,
        cols: usize,
        rhs_grad: &mut Buffer<T, D, RS>,
        out_grad: &Buffer<T, D, LS>,
    );
}

pub trait RandOp<T, S: Shape = (), D: Device = Self>: Device {
    fn rand(&self, x: &mut Buffer<T, D, S>, lo: T, hi: T);
}

pub trait Softmax<T, S: Shape = (), D: Device = Self>: Device {
    fn softmax(&self, samples: usize, features: usize, x: &Buffer<T, D, S>) -> Buffer<T, Self, S>;
}

pub trait SoftmaxGrad<T, S: Shape = ()>: Device {
    fn softmax_grad(
        &self,
        samples: usize,
        features: usize,
        x: &Buffer<T, Self, S>,
        x_grad: &Buffer<T, Self, S>,
        out_grad: &Buffer<T, Self, S>,
    );
}

//pub trait SumOp
pub trait Max<T, S: Shape = (), D: Device = Self>: Device {
    fn max(&self, x: &Buffer<T, D, S>) -> T;
}

pub trait MaxRows<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn max_rows(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub trait MaxRowsGrad<T, IS: Shape = (), OS: Shape = ()>: Device {
    fn max_rows_grad(
        &self,
        cols: usize,
        out: &Buffer<T, Self, OS>,
        x: &Buffer<T, Self, IS>,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    );
}

pub trait MaxCols<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn max_cols(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub trait MaxColsGrad<T, IS: Shape = (), OS: Shape = ()>: Device {
    fn max_cols_grad(
        &self,
        cols: usize,
        out: &Buffer<T, Self, OS>,
        x: &Buffer<T, Self, IS>,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    );
}

pub trait Sum<T, S: Shape = (), D: Device = Self>: Device {
    fn sum(&self, x: &Buffer<T, D, S>) -> T;
}

pub trait SumRows<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn sum_rows(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub trait SumRowsGrad<T, IS: Shape = (), OS: Shape = ()>: Device {
    fn sum_rows_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, IS>,
    );
}

pub trait SumCols<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn sum_cols(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub trait SumColsGrad<T, IS: Shape = (), OS: Shape = ()>: Device {
    fn sum_cols_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, IS>,
    );
}

/// Provides the onehot encoding operation.
pub trait Onehot<T, IS: Shape = (), OS: Shape = ()>: Device {
    /// Onehot encodes a `Buffer` of classes.
    ///
    /// # Example
    ///
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    ///
    /// use sliced::{CPU, Onehot, Buffer};
    ///
    /// let device = CPU::new();
    ///
    /// let classes = Buffer::from((&device, [1, 0, 3, 2]));
    /// let onehot = device.onehot(&classes);
    ///
    /// assert_eq!([
    ///     0, 1, 0, 0,
    ///     1, 0, 0, 0,
    ///     0, 0, 0, 1,
    ///     0, 0, 1, 0
    /// ], &*onehot);
    /// ```
    fn onehot(&self, classes: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

/// Calculates the diagflat of an 1D [`Buffer`] without gradients.
pub trait Diagflat<T, IS: Shape = (), OS: Shape = ()>: Device {
    /// Takes the values of [`Buffer`] `x` and puts them diagonally on the `Buffer` `out`.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use sliced::{Diagflat, Buffer, CPU};
    ///
    /// let device = CPU::new();
    /// let x = Buffer::from((&device, [1, 2, 7, -1, -2]));
    /// let out = device.diagflat(&x);
    ///
    /// assert_eq!(&*out, [
    ///     1, 0, 0, 0, 0,
    ///     0, 2, 0, 0, 0,
    ///     0, 0, 7, 0, 0,
    ///     0, 0, 0, -1, 0,
    ///     0, 0, 0, 0, -2,
    /// ]);
    ///
    /// ```
    fn diagflat(&self, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

pub trait DiagflatGrad<T, IS: Shape = (), OS: Shape = ()>: Device {
    fn diagflat_grad(
        &self,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    );
}

pub trait Mean<T, S: Shape>: Device {
    fn mean(&self, x: &Buffer<T, Self, S>) -> T;
}

/// Calculates the mean of every column (while interacting with the rows).
pub trait MeanRows<T, IS: Shape = (), OS: Shape = ()>: Device {
    /// Calculates the mean of every column (while interacting with the rows).
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use sliced::{CPU, MeanRows, Buffer};
    ///
    /// let device = CPU::new();
    ///
    /// let to_mean_rows = Buffer::from((&device, [
    ///     2, 1, 3,
    ///     1, 2, 3,
    ///     3, 1, 1,
    ///     2, 4, 1,
    /// ]));
    /// let mean_rows: Buffer<_> = device.mean_rows(3, &to_mean_rows);
    /// assert_eq!(&*mean_rows, [2, 2, 2]);
    ///
    /// ```
    fn mean_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

pub trait MeanRowsGrad<T, IS: Shape = (), OS: Shape = ()>: Device {
    fn mean_rows_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    );
}

/// Calculates the mean of every row (while interacting with the columns).
pub trait MeanCols<T, IS: Shape = (), OS: Shape = ()>: Device {
    /// Calculates the mean of every row (while interacting with the columns).
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use sliced::{Buffer, CPU, MeanCols};
    ///
    /// let device = CPU::new();
    ///
    /// let to_mean_cols = Buffer::from((&device, [
    ///     1, 4, 1, 2,
    ///     2, 2, 2, 2,
    ///     8, 1, 2, 1,
    /// ]));
    ///
    /// let mean_cols: Buffer<_>  = device.mean_cols(4, &to_mean_cols);
    /// assert_eq!(&*mean_cols, [2, 2, 3])
    /// ```
    fn mean_cols(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS>;
}

pub trait MeanColsGrad<T, IS: Shape = (), OS: Shape = ()>: Device {
    fn mean_cols_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    );
}
