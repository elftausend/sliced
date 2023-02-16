use custos::{Buffer, Device, Eval, MayDim2, Resolve, Shape};

mod cpu;

#[cfg(feature = "cpu")]
pub use cpu::*;

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

pub trait BinaryEWWithGrad<T, S: Shape = (), D: Device = Self>: Device {
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
    fn softmax(&self, samples: usize, x: &Buffer<T, D, S>) -> Buffer<T, D, S>;
}

//pub trait SumOp
pub trait Max<T, S: Shape = (), D: Device = Self>: Device {
    fn max(&self, x: &Buffer<T, D, S>) -> T;
}

pub trait MaxRows<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn max_rows(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub trait MaxCols<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn max_cols(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub trait Sum<T, S: Shape = (), D: Device = Self>: Device {
    fn sum(&self, x: &Buffer<T, D, S>) -> T;
}

pub trait SumRows<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn sum_rows(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}

pub trait SumCols<T, IS: Shape = (), OS: Shape = (), D: Device = Self>: Device {
    fn sum_cols(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS>;
}
