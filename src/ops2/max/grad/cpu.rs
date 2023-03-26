use std::ops::AddAssign;

use custos::{prelude::One, Buffer, CPU};

use crate::{MaxColsGrad, MaxRowsGrad};

impl<T: PartialEq + Copy + AddAssign> MaxRowsGrad<T> for CPU {
    #[inline]
    fn max_rows_grad(
        &self,
        cols: usize,
        out: &Buffer<T>,
        x: &Buffer<T>,
        x_grad: &mut Buffer<T>,
        out_grad: &Buffer<T>,
    ) {
        slice_max_rows_grad(cols, out, x, x_grad, out_grad);
    }
}

impl<T: PartialEq + Copy + AddAssign> MaxColsGrad<T> for CPU {
    #[inline]
    fn max_cols_grad(
        &self,
        cols: usize,
        out: &Buffer<T>,
        x: &Buffer<T>,
        x_grad: &mut Buffer<T>,
        out_grad: &Buffer<T>,
    ) {
        slice_max_cols_grad(cols, out, x, x_grad, out_grad);
    }
}

#[inline]
pub fn max_grad<T: PartialEq + One>(out: &T, x: &[T], x_grad: &mut [T]) {
    x_grad[x.iter().position(|val| val == out).unwrap()] = T::one()
}

pub fn slice_max_rows_grad<T>(cols: usize, out: &[T], x: &[T], x_grad: &mut [T], out_grad: &[T])
where
    T: PartialEq + Copy + AddAssign,
{
    let rows = x.len() / cols;

    for (col, out_val) in out.iter().enumerate() {
        for row in 0..rows {
            let grad_idx = row * cols + col;

            if out_val == &x[grad_idx] {
                x_grad[grad_idx] += out_grad[col];
            }
        }
    }
}

pub fn slice_max_cols_grad<T>(cols: usize, out: &[T], x: &[T], x_grad: &mut [T], out_grad: &[T])
where
    T: PartialEq + Copy + AddAssign,
{
    for (idx, ((row, max), grad)) in x.chunks(cols).zip(out).zip(out_grad).enumerate() {
        let grad_idx = idx * cols + row.iter().position(|val| val == max).expect("Could not find maximum in gradient calculation");
        x_grad[grad_idx] += *grad
    }
}

#[cfg(test)]
mod tests {
    use crate::{max_cols, slice_max_cols_grad, max_grad, max_rows, slice_max_rows_grad};

    #[test]
    fn test_max_cols_grad() {
        #[rustfmt::skip]
        let x = [-3, 2, 3, 1,
                            1, 5, -5, 4,
                            -9, -2, -4, -1];

        let mut out = [0; 3];
        max_cols(4, &x, &mut out);

        let mut x_grad = [0; 12];

        let out_grad = [1, 2, 3];

        slice_max_cols_grad(4, &out, &x, &mut x_grad, &out_grad);

        #[rustfmt::skip]
        let expected = [
            0, 0, 1, 0,
            0, 2, 0, 0,
            0, 0, 0, 3,
        ];

        assert_eq!(expected, x_grad);
    }

    #[test]
    fn test_max_rows_grad() {
        #[rustfmt::skip]
        let x = [-3, 2, 3, 1,
                            1, 5, -5, 4,
                            -9, -2, -4, -1];

        let mut out = [0; 4];
        max_rows(4, &x, &mut out);

        let mut x_grad = [0; 12];

        let out_grad = [2, 3, 4, 1];

        slice_max_rows_grad(4, &out, &x, &mut x_grad, &out_grad);

        #[rustfmt::skip]
        let expected = [
            0, 0, 4, 0,
            2, 3, 0, 1,
            0, 0, 0, 0,
        ];

        assert_eq!(expected, x_grad);
    }

    #[test]
    pub fn test_max_grad() {
        #[rustfmt::skip]
        let x = [-3, 2, 3, 1,
                            1, 5, -5, 4,
                            -9, -2, -4, -1];

        let mut x_grad = [0; 12];
        max_grad(&5, &x, &mut x_grad);

        #[rustfmt::skip]
        let expected = [
            0, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 0,
        ];
        assert_eq!(expected, x_grad);
    }
}
