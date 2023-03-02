use std::ops::{Add, AddAssign};

use custos::{Alloc, Buffer, MainMemory, Shape, CPU};

use crate::RowOp;

impl<T: Add<Output = T> + Copy + AddAssign, LS: Shape, RS: Shape> RowOp<T, LS, RS> for CPU {
    #[inline]
    fn add_row(
        &self,
        rows: usize,
        cols: usize,
        lhs: &Buffer<T, Self, LS>,
        rhs: &Buffer<T, Self, RS>,
    ) -> Buffer<T, Self, LS> {
        row_op(self, rows, cols, lhs, rhs, |c, a, b| *c = a + b)
    }

    #[inline]
    fn add_row_mut(
        &self,
        rows: usize,
        cols: usize,
        lhs: &mut Buffer<T, Self, LS>,
        rhs: &Buffer<T, Self, RS>,
    ) {
        row_op_slice_lhs(lhs, rows, cols, rhs, |c, a| *c += a)
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
    row_op_slice_mut(lhs, rows, cols, rhs, &mut out, f);
    out
}

pub fn row_op_slice_lhs<T, F>(lhs: &mut [T], lhs_rows: usize, lhs_cols: usize, rhs: &[T], f: F)
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

pub fn row_op_slice_mut<T, F>(lhs: &[T], lrows: usize, lcols: usize, rhs: &[T], out: &mut [T], f: F)
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
