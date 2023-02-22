use custos::{prelude::One, Buffer, Device, MainMemory, Shape, CPU};

use crate::{Max, MaxCols, MaxColsGrad, MaxRows, MaxRowsGrad};

impl<T, D, S> Max<T, S, D> for CPU
where
    T: Ord + Copy,
    D: MainMemory,
    S: Shape,
{
    #[inline]
    fn max(&self, x: &Buffer<T, D, S>) -> T {
        max(x).expect("Buffer should contain at least an element.")
    }
}

impl<T, D, IS, OS> MaxRows<T, IS, OS, D> for CPU
where
    T: Ord + Copy,
    D: MainMemory,
    IS: Shape,
    OS: Shape,
{
    #[inline]
    fn max_rows(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve::<_, OS>(cols);

        // If all values in a column are negative, the corresponding maximum would be 0.
        out.copy_from_slice(&x[..cols]);

        max_rows(cols, x, &mut out);
        out
    }
}

impl<T: PartialEq + Copy> MaxRowsGrad<T> for CPU {
    #[inline]
    fn max_rows_grad(
        &self,
        cols: usize,
        out: &Buffer<T>,
        x: &Buffer<T>,
        x_grad: &mut Buffer<T>,
        out_grad: &Buffer<T>,
    ) {
        max_rows_grad(cols, out, x, x_grad, out_grad);
    }
}

impl<T, D> MaxCols<T, (), (), D> for CPU
where
    T: Ord + Copy,
    D: MainMemory,
{
    #[inline]
    fn max_cols(&self, rows: usize, cols: usize, x: &Buffer<T, D>) -> Buffer<T> {
        let mut out = self.retrieve::<_, ()>(rows);
        max_cols(cols, x, &mut out);
        out
    }
}

impl<T: PartialEq + Copy> MaxColsGrad<T> for CPU {
    #[inline]
    fn max_cols_grad(
        &self,
        cols: usize,
        out: &Buffer<T, Self, ()>,
        x: &Buffer<T, Self, ()>,
        x_grad: &mut Buffer<T, Self, ()>,
        out_grad: &Buffer<T, Self, ()>,
    ) {
        max_cols_grad(cols, out, x, x_grad, out_grad);
    }
}

#[inline]
pub fn max<T: Ord + Copy>(x: &[T]) -> Option<T> {
    x.iter().copied().reduce(T::max)
}

#[inline]
pub fn max_grad<T: PartialEq + One>(out: &T, x: &[T], x_grad: &mut [T]) {
    x_grad[x.iter().position(|val| val == out).unwrap()] = T::one()
}

pub fn max_rows<T: Ord + Copy>(cols: usize, x: &[T], out: &mut [T]) {
    for row in x.chunks(cols) {
        for (val, max) in row.iter().zip(out.iter_mut()) {
            *max = T::max(*val, *max)
        }
    }
}

pub fn max_rows_grad<T>(cols: usize, out: &[T], x: &[T], x_grad: &mut [T], out_grad: &[T])
where
    T: PartialEq + Copy,
{
    let rows = x.len() / cols;

    for (col, out_val) in out.iter().enumerate() {
        for row in 0..rows {
            let grad_idx = row * cols + col;

            if out_val == &x[grad_idx] {
                x_grad[grad_idx] = out_grad[col];
            }
        }
    }
}

pub fn max_cols<T: Ord + Copy>(cols: usize, x: &[T], out: &mut [T]) {
    for (row, val) in x.chunks(cols).zip(out) {
        *val = max(row).expect("The slice should contain at least one value.");
    }
}

pub fn max_cols_grad<T>(cols: usize, out: &[T], x: &[T], x_grad: &mut [T], out_grad: &[T])
where
    T: PartialEq + Copy,
{
    for (idx, ((row, max), grad)) in x.chunks(cols).zip(out).zip(out_grad).enumerate() {
        let grad_idx = idx * cols + row.iter().position(|val| val == max).unwrap();
        x_grad[grad_idx] = *grad
    }
}

#[cfg(test)]
mod tests {
    use crate::{max_cols, max_cols_grad, max_grad, max_rows, max_rows_grad};

    #[test]
    fn test_max_rows() {
        #[rustfmt::skip]
        let x = [-3, 2, 3, 1,
                            1, 5, -5, 4,
                            -9, -2, -4, -1];

        let mut out = [0; 4];
        max_rows(4, &x, &mut out);
        assert_eq!(out, [1, 5, 3, 4]);
    }

    #[test]
    fn test_max_rows_neg() {
        #[rustfmt::skip]
        let x = [-3, -2, -3, -1,
                            -1, -5, -5, -4,
                            -9, -2, -4, -1];

        let mut out = [-3, -2, -3, -1];
        max_rows(4, &x, &mut out);
        assert_eq!(out, [-1, -2, -3, -1]);
    }

    #[test]
    fn test_max_cols() {
        #[rustfmt::skip]
        let x = [-3, 2, 3, 1,
                            1, 5, -5, 4,
                            -9, -2, -4, -1];

        let mut out = [0; 3];
        max_cols(4, &x, &mut out);
        assert_eq!(out, [3, 5, -1]);
    }

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

        max_cols_grad(4, &out, &x, &mut x_grad, &out_grad);

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

        max_rows_grad(4, &out, &x, &mut x_grad, &out_grad);

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
