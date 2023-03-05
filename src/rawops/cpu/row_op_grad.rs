use std::ops::AddAssign;

use custos::{Buffer, Shape, CPU};

use crate::{slice_sum_rows, RowOpGrad};

impl<T, LS, RS> RowOpGrad<T, LS, RS> for CPU
where
    T: Copy + AddAssign,
    LS: Shape,
    RS: Shape,
{
    #[inline]
    fn add_row_grad(
        &self,
        rows: usize,
        cols: usize,
        lhs_grad: &mut Buffer<T, Self, LS>,
        rhs_grad: &mut Buffer<T, Self, RS>,
        out_grad: &Buffer<T, Self, LS>,
    ) {
        add_row_op_grad(rows, cols, lhs_grad, rhs_grad, out_grad);
    }

    #[inline]
    fn add_row_mut_grad(
        &self,
        rows: usize,
        cols: usize,
        rhs_grad: &mut Buffer<T, Self, RS>,
        out_grad: &Buffer<T, Self, LS>,
    ) {
        slice_sum_rows(rows, cols, out_grad, rhs_grad);
    }
}

#[inline]
pub fn add_row_op_grad<T: Copy + AddAssign>(
    rows: usize,
    cols: usize,
    lhs_grad: &mut [T],
    rhs_grad: &mut [T],
    out_grad: &[T],
) {
    lhs_grad.copy_from_slice(out_grad);

    slice_sum_rows(rows, cols, out_grad, rhs_grad);
}
