use std::{iter::Sum, ops::AddAssign};

use custos::{Buffer, Device, MainMemory, Shape, CPU};

use crate::{SumCols, SumColsGrad, SumRows, SumRowsGrad};

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

impl<T, IS, OS> SumRowsGrad<T, IS, OS> for CPU
where
    T: Copy + AddAssign,
    IS: Shape,
    OS: Shape,
{
    #[inline]
    fn sum_rows_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, IS>,
    ) {
        sum_rows_grad(cols, x_grad, out_grad);
    }
}

impl<T, IS, OS> SumColsGrad<T, IS, OS> for CPU
where
    T: Copy + AddAssign,
    IS: Shape,
    OS: Shape,
{
    #[inline]
    fn sum_cols_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, IS>,
    ) {
        sum_cols_grad(cols, x_grad, out_grad);
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
    fn sum_cols(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS> {
        let rows = x.len() / cols;
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

/// Accumulates to out
pub fn sum_rows2<T: AddAssign + Copy>(cols: usize, x: &[T], out: &mut [T]) {
    for row in x.chunks(cols) {
        for (val, out) in row.iter().zip(&mut *out) {
            *out += *val;
        }
    }
}

pub fn sum_rows_grad<T: Copy + AddAssign>(cols: usize, x_grad: &mut [T], out_grad: &[T]) {
    for x_grad in x_grad.chunks_mut(cols) {
        for (x, out) in x_grad.iter_mut().zip(out_grad) {
            *x += *out;
        }
        //x_grad.copy_from_slice(out_grad)
    }
}

pub fn sum_cols<T: Sum<T> + Copy>(cols: usize, x: &[T], out: &mut [T]) {
    for (row, out) in x.chunks(cols).zip(out) {
        *out = row.iter().copied().sum::<T>();
    }
}

pub fn sum_cols_grad<T: Copy + AddAssign>(cols: usize, x_grad: &mut [T], out_grad: &[T]) {
    for (x_grad, out_grad) in x_grad.chunks_mut(cols).zip(out_grad) {
        for val in x_grad {
            *val += *out_grad
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{sum_cols, sum_cols_grad, sum_rows2, sum_rows_grad};

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

    #[test]
    fn test_sum_cols_grad() {
        #[rustfmt::skip]
        let _x = [
            1, 2, 3,
            3, 2, 1,
            2, 3, 1,
            4, 8, 1
        ];
        let mut x_grad = [0; 12];
        sum_cols_grad(3, &mut x_grad, &[2, 3, 4, -1]);

        #[rustfmt::skip]
        let expected = [
            2, 2, 2, 
            3, 3, 3, 
            4, 4, 4, 
            -1, -1, -1
        ];
        assert_eq!(x_grad, expected);
    }

    #[test]
    fn test_sum_rows_grad() {
        #[rustfmt::skip]
        let _x = [
            1, 2, 3,
            3, 2, 1,
            2, 3, 1,
            4, 8, 1
        ];
        let mut x_grad = [0; 12];
        sum_rows_grad(3, &mut x_grad, &[2, 4, -1]);

        #[rustfmt::skip]
        let expected = [
            2, 4, -1, 
            2, 4, -1, 
            2, 4, -1, 
            2, 4, -1
        ];
        assert_eq!(x_grad, expected);
    }
}
