use std::{
    iter::Sum,
    ops::{AddAssign, Deref},
};

use custos::{AddOperation, AsNoId, Buffer, Device, OnDropBuffer, Retrieve, Retriever, Shape, CPU};

use crate::{SumCols, SumRows};

impl<T, S, D, Mods: OnDropBuffer> crate::Sum<T, S, D> for CPU<Mods>
where
    T: Copy + Sum,
    S: Shape,
    D: Device,
    D::Base<T, S>: Deref<Target = [T]>,
{
    #[inline]
    fn sum(&self, x: &Buffer<T, D, S>) -> T {
        x.iter().copied().sum()
    }
}

impl<T, IS, OS, D, Mods> SumRows<T, IS, OS, D> for CPU<Mods>
where
    T: Copy + Sum + AddAssign + 'static,
    IS: Shape,
    OS: Shape,
    D: Device + 'static,
    D::Base<T, IS>: Deref<Target = [T]>,
    Mods: Retrieve<Self, T, OS> + AddOperation + 'static,
{
    #[inline]
    fn sum_rows(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve(cols, x).unwrap();
        self.add_op((cols.no_id(), x, &mut out), |(cols, x, out)| {
            slice_sum_rows2(**cols, x, out);
            Ok(())
        })
        .unwrap();
        out
    }
}

impl<T, IS, OS, D, Mods> SumCols<T, IS, OS, D> for CPU<Mods>
where
    T: Copy + Sum + 'static,
    IS: Shape,
    OS: Shape,
    D: Device + 'static,
    D::Base<T, IS>: Deref<Target = [T]>,
    Mods: Retrieve<Self, T, OS> + AddOperation + 'static,
{
    #[inline]
    fn sum_cols(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS> {
        let rows = x.len() / cols;
        let mut out = self.retrieve(rows, x).unwrap();
        self.add_op((cols.no_id(), x, &mut out), |(cols, x, out)| {
            slice_sum_cols(**cols, x, out);
            Ok(())
        })
        .unwrap();
        out
    }
}

pub fn slice_sum_rows<T: AddAssign + Copy>(rows: usize, cols: usize, x: &[T], out: &mut [T]) {
    for idx in 0..rows {
        let index = idx * cols;
        let row = &x[index..index + cols];

        for (i, value) in row.iter().enumerate() {
            out[i] += *value;
        }
    }
}

/// Accumulates to out
pub fn slice_sum_rows2<T: AddAssign + Copy>(cols: usize, x: &[T], out: &mut [T]) {
    for row in x.chunks(cols) {
        for (val, out) in row.iter().zip(&mut *out) {
            *out += *val;
        }
    }
}

pub fn slice_sum_cols<T: Sum<T> + Copy>(cols: usize, x: &[T], out: &mut [T]) {
    for (row, out) in x.chunks(cols).zip(out) {
        *out = row.iter().copied().sum::<T>();
    }
}

#[cfg(test)]
mod tests {
    use crate::{slice_sum_cols, slice_sum_rows2};

    #[test]
    fn test_sum_rows() {
        #[rustfmt::skip]
        let x = [
            1, 2, 3,
            3, 2, 1,
            2, 3, 1,
            4, 8, 1
        ];

        let mut out = [0, 0, 0];
        slice_sum_rows2(3, &x, &mut out);
        assert_eq!(out, [10, 15, 6]);
    }
    #[test]
    fn test_sum_cols() {
        #[rustfmt::skip]
        let x = [
            1, 2, 3,
            3, 2, 1,
            2, 3, 1,
            4, 8, 1
        ];

        let mut out = [0, 0, 0, 0];
        slice_sum_cols(3, &x, &mut out);
        assert_eq!(out, [6, 6, 6, 13]);
    }
}
