use std::{iter::Sum, ops::AddAssign};

use custos::{MainMemory, Shape, CPU, Buffer, Device};

use crate::{SumRows, SumCols};

impl<T, S, D> crate::Sum<T, S, D> for CPU
where
    T: Copy + Sum,
    S: Shape,
    D: MainMemory,
{
    #[inline]
    fn sum(&self, x: &Buffer<T, D, S>) -> T {
        x.iter().copied().sum()
    }
}

impl<T, IS, OS, D> SumRows<T, IS, OS, D> for CPU
where
    T: Copy + Sum + AddAssign,
    IS: Shape,
    OS: Shape,
    D: MainMemory,
{
    #[inline]
    fn sum_rows(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve(cols);
        sum_rows2(cols, x, &mut out);
        out
    }
}

impl<T, IS, OS, D> SumCols<T, IS, OS, D> for CPU
where
    T: Copy + Sum,
    IS: Shape,
    OS: Shape,
    D: MainMemory,
{
    #[inline]
    fn sum_cols(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve(rows);
        sum_cols(cols, x, &mut out);
        out
    }
}

pub fn sum_rows<T: AddAssign + Copy>(rows: usize, cols: usize, x: &[T], out: &mut [T]) {
    for idx in 0..rows {
        let index = idx * cols;
        let row = &x[index..index + cols];

        for (i, value) in row.iter().enumerate() {
            out[i] += *value;
        }
    }
}

pub fn sum_rows2<T: AddAssign + Copy>(cols: usize, x: &[T], out: &mut [T]) {
    for row in x.chunks(cols) {
        for (val, out) in row.iter().zip(&mut *out) {
            *out += *val;
        }
    }
}

pub fn sum_cols<T: Sum<T> + Copy>(cols: usize, x: &[T], out: &mut [T]) {
    for (row, out) in x.chunks(cols).zip(out) {
        *out = row.iter().copied().sum::<T>();
    }
}

#[cfg(test)]
mod tests {
    use crate::{sum_cols, sum_rows2};

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
        sum_rows2(3, &x, &mut out);
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
        sum_cols(3, &x, &mut out);
        assert_eq!(out, [6, 6, 6, 13]);
    }
}
