use std::ops::{Add, AddAssign};

use custos::{Alloc, Buffer, MainMemory, Shape, CPU};

use crate::RowOp;

impl<T, LS, RS> RowOp<T, LS, RS> for CPU
where
    T: Add<Output = T> + Copy + AddAssign,
    LS: Shape,
    RS: Shape,
{
    #[inline]
    fn add_row_mut(
        &self,
        rows: usize,
        cols: usize,
        lhs: &mut Buffer<T, Self, LS>,
        rhs: &Buffer<T, Self, RS>,
    ) {
        slice_row_op_lhs(rows, cols, lhs, rhs, |c, a| *c += a)
    }

    #[inline]
    fn row_op<F: Fn(&mut T, T, T)>(
        &self,
        rows: usize,
        cols: usize,
        lhs: &Buffer<T, Self, LS>,
        rhs: &Buffer<T, Self, RS>,
        f: F,
    ) -> Buffer<T, Self, LS> {
        row_op(self, rows, cols, lhs, rhs, f)
    }
}

pub fn row_op<'a, T, F, D, Host, LS: Shape, RS: Shape>(
    device: &'a Host,
    rows: usize,
    cols: usize,
    lhs: &Buffer<T, D, LS>,
    rhs: &Buffer<T, D, RS>,
    f: F,
) -> Buffer<'a, T, Host, LS>
where
    T: Copy,
    F: Fn(&mut T, T, T),
    D: MainMemory,
    Host: for<'b> Alloc<'b, T, LS> + MainMemory,
{
    debug_assert_eq!(rhs.len(), cols);

    let mut out = device.retrieve(lhs.len(), (lhs, rhs));
    slice_row_op_mut(rows, cols, lhs, rhs, &mut out, f);
    out
}

pub fn slice_row_op_lhs<T, F>(lhs_rows: usize, lhs_cols: usize, lhs: &mut [T], rhs: &[T], f: F)
where
    T: Copy,
    F: Fn(&mut T, T),
{
    for i in 0..lhs_rows {
        let index = i * lhs_cols;

        for (idx, value) in rhs.iter().enumerate() {
            f(&mut lhs[index + idx], *value);
        }
    }
}

pub fn slice_row_op_mut<T, F>(lrows: usize, lcols: usize, lhs: &[T], rhs: &[T], out: &mut [T], f: F)
where
    T: Copy,
    F: Fn(&mut T, T, T),
{
    for i in 0..lrows {
        let index = i * lcols;
        let x = &lhs[index..index + lcols];

        for (idx, value) in rhs.iter().enumerate() {
            f(&mut out[index + idx], x[idx], *value);
        }
    }
}
