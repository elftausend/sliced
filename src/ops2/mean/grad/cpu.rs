use std::ops::{AddAssign, Mul};

use crate::{MeanColsGrad, MeanRowsGrad};
use custos::{prelude::Number, Buffer, OnDropBuffer, Shape, CPU};

impl<T: Number, IS: Shape, OS: Shape, Mods: OnDropBuffer> MeanRowsGrad<T, IS, OS> for CPU<Mods> {
    #[inline]
    fn mean_rows_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    ) {
        mean_rows_grad(cols, x_grad, out_grad);
    }
}

impl<T: Number, IS: Shape, OS: Shape, Mods: OnDropBuffer> MeanColsGrad<T, IS, OS> for CPU<Mods> {
    #[inline]
    fn mean_cols_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    ) {
        mean_cols_grad(cols, x_grad, out_grad);
    }
}

pub fn slice_rows_grad_unary<T: Mul<Output = T> + AddAssign + Copy>(
    cols: usize,
    x_grad: &mut [T],
    out_grad: &[T],
    x_grad_fn: impl Fn(&T) -> T,
) {
    for x_grad in x_grad.chunks_mut(cols) {
        for (x, out) in x_grad.iter_mut().zip(out_grad) {
            *x += x_grad_fn(x) * *out;
        }
    }
}

pub fn mean_rows_grad<T: Number>(cols: usize, x_grad: &mut [T], out_grad: &[T]) {
    let len = x_grad.len();
    slice_rows_grad_unary(cols, x_grad, out_grad, |_| {
        T::from_usize(cols) / (T::from_usize(len))
    });

    /*for x_grad in x_grad.chunks_mut(cols) {
        for (x, out) in x_grad.iter_mut().zip(out_grad) {
            *x += *out / T::from_usize(len / cols);
        }
    }*/
}

pub fn mean_cols_grad<T: Number>(cols: usize, x_grad: &mut [T], out_grad: &[T]) {
    for (row, out) in x_grad.chunks_mut(cols).zip(out_grad) {
        let div = *out / T::from_usize(cols);
        for val in row {
            *val += div;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{mean_cols_grad, mean_rows_grad};

    #[test]
    fn test_mean_rows_grad() {
        #[rustfmt::skip]
        let mut x_grad = [0.; 12];

        mean_rows_grad(4, &mut x_grad, &[9., -3., 6., 3.]);

        #[rustfmt::skip]
        let expected = [
            3.0, -1.0, 2.0, 1.0, 
            3.0, -1.0, 2.0, 1.0, 
            3.0, -1.0, 2.0, 1.0
        ];

        assert_eq!(expected, x_grad);
    }

    #[test]
    fn test_mean_cols_grad() {
        let mut x_grad = [0.; 12];

        mean_cols_grad(4, &mut x_grad, &[2., -1., 3.]);

        #[rustfmt::skip]
        let expected = [
            0.5, 0.5, 0.5, 0.5, 
            -0.25, -0.25, -0.25, -0.25, 
            0.75, 0.75, 0.75, 0.75
        ];

        assert_eq!(expected, x_grad);
    }
}
