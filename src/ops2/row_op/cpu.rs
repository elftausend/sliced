use std::ops::{Add, AddAssign, Deref};

use custos::{Alloc, Buffer, Device, Shape, CPU, Retriever, Retrieve};

use crate::RowOp;

impl<T, LS, RS, Mods: Retrieve<Self, T>> RowOp<T, LS, RS> for CPU<Mods>
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
        cols: usize,
        lhs: &Buffer<T, Self, LS>,
        rhs: &Buffer<T, Self, RS>,
        f: F,
    ) -> Buffer<T, Self, LS> {
        row_op(self, cols, lhs, rhs, f)
    }
}

pub fn row_op<'a, T, F, D, Host, LS: Shape, RS: Shape>(
    device: &'a Host,
    cols: usize,
    lhs: &Buffer<T, D, LS>,
    rhs: &Buffer<T, D, RS>,
    f: F,
) -> Buffer<'a, T, Host, LS>
where
    T: Copy,
    F: Fn(&mut T, T, T),
    D: Device,
    D::Data<T, LS>: Deref<Target = [T]>,
    D::Data<T, RS>: Deref<Target = [T]>,
    Host: Alloc<T> + Retriever<T>,
    Host::Data<T, LS>: Deref<Target = [T]>,
{
    debug_assert_eq!(rhs.len(), cols);

    let mut out = device.retrieve(lhs.len(), (lhs, rhs));
    slice_row_op_mut(cols, lhs, rhs, &mut out, f);
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

pub fn slice_row_op_mut<T, F>(cols: usize, lhs: &[T], rhs: &[T], out: &mut [T], f: F)
where
    T: Copy,
    F: Fn(&mut T, T, T),
{
    for (row, out) in lhs.chunks(cols).zip(out.chunks_mut(cols)) {
        for ((lhs, rhs), out) in row.iter().zip(rhs).zip(out) {
            f(out, *lhs, *rhs);
        }
    }
    /*for i in 0..lrows {
        let index = i * lcols;
        let x = &lhs[index..index + lcols];

        for (idx, value) in rhs.iter().enumerate() {
            f(&mut out[index + idx], x[idx], *value);
        }
    }*/
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_row_op_slice() {
        // 3 x 4
        let lhs = [4, 3, 2, 3, 5, 7, 1, 1, 7, 2, 3, 4];

        let rhs = [1, 2, 3, 4];

        let mut out = [0; 12];

        super::slice_row_op_mut(4, &lhs, &rhs, &mut out, |out, lhs, rhs| *out = lhs + rhs);

        assert_eq!(out, [5, 5, 5, 7, 6, 9, 4, 5, 8, 4, 6, 8]);
    }
}
