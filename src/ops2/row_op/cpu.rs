use std::ops::{Add, AddAssign, Deref, DerefMut};

use custos::{Alloc, Buffer, Device, Retrieve, Retriever, Shape, CPU, Resolve, MayToCLSource, Eval, ToVal};

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
    fn row_op<O: Eval<T> + MayToCLSource>(
        &self,
        cols: usize,
        lhs: &Buffer<T, Self, LS>,
        rhs: &Buffer<T, Self, RS>,
        f: impl Fn(Resolve<T>, Resolve<T>) -> O,
    ) -> Buffer<T, Self, LS> {
        row_op(self, cols, lhs, rhs, f)
    }
}

pub fn row_op<'a, T, O, D, Host, LS: Shape, RS: Shape>(
    device: &'a Host,
    cols: usize,
    lhs: &Buffer<T, D, LS>,
    rhs: &Buffer<T, D, RS>,
    f: impl Fn(Resolve<T>, Resolve<T>) -> O,
) -> Buffer<'a, T, Host, LS>
where
    T: Copy,
    O: Eval<T> + MayToCLSource,
    D: Device,
    D::Data<T, LS>: Deref<Target = [T]>,
    D::Data<T, RS>: Deref<Target = [T]>,
    Host: Alloc<T> + Retriever<T>,
    Host::Data<T, LS>: Deref<Target = [T]> + DerefMut,
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

pub fn slice_row_op_mut<T, O>(cols: usize, lhs: &[T], rhs: &[T], out: &mut [T], f: impl Fn(Resolve<T>, Resolve<T>) -> O)
where
    T: Copy,
    O: Eval<T>
{
    for (row, out) in lhs.chunks(cols).zip(out.chunks_mut(cols)) {
        for ((lhs, rhs), out) in row.iter().zip(rhs).zip(out) {
            *out = f((*lhs).to_val(), (*rhs).to_val()).eval();
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
    use custos::Combiner;

    #[test]
    fn test_row_op_slice() {
        // 3 x 4
        #[rustfmt::skip]
        let lhs = [
            4, 3, 2, 3, 
            5, 7, 1, 1, 
            7, 2, 3, 4
        ];

        let rhs = [1, 2, 3, 4];

        let mut out = [0; 12];

        super::slice_row_op_mut(4, &lhs, &rhs, &mut out, |lhs, rhs| lhs.add(rhs));

        assert_eq!(out, [5, 5, 5, 7, 6, 9, 4, 5, 8, 4, 6, 8]);
    }
}
