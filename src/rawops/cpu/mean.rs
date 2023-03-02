use custos::{prelude::Number, Buffer, Device, Shape, CPU};

use crate::{sum_rows2, Mean, MeanCols, MeanColsGrad, MeanRows, MeanRowsGrad};

impl<T: Number, IS: Shape> Mean<T, IS> for CPU {
    #[inline]
    fn mean(&self, x: &Buffer<T, Self, IS>) -> T {
        mean(x)
    }
}

impl<T: Number, IS: Shape, OS: Shape> MeanRows<T, IS, OS> for CPU {
    #[inline]
    fn mean_rows(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve(cols, x);
        mean_rows(cols, x, &mut out);
        out
    }
}

impl<T: Number, IS: Shape, OS: Shape> MeanCols<T, IS, OS> for CPU {
    #[inline]
    fn mean_cols(&self, cols: usize, x: &Buffer<T, Self, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve(x.len() / cols, x);
        mean_cols(cols, x, &mut out);
        out
    }
}

impl<T: Number, IS: Shape, OS: Shape> MeanRowsGrad<T, IS, OS> for CPU {
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

impl<T: Number, IS: Shape, OS: Shape> MeanColsGrad<T, IS, OS> for CPU {
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

pub fn mean<T: Number>(x: &[T]) -> T {
    x.iter().copied().sum::<T>() / T::from_usize(x.len())
}

pub fn mean_rows<T: Number>(cols: usize, x: &[T], out: &mut [T]) {
    sum_rows2(cols, x, out);
    for val in out {
        *val /= T::from_usize(x.len() / cols)
    }
}

pub fn mean_rows_grad<T: Number>(cols: usize, x_grad: &mut [T], out_grad: &[T]) {
    let len = x_grad.len();
    for x_grad in x_grad.chunks_mut(cols) {
        for (x, out) in x_grad.iter_mut().zip(out_grad) {
            *x += *out / T::from_usize(len / cols);
        }
    }
}

pub fn mean_cols<T: Number>(cols: usize, x: &[T], out: &mut [T]) {
    for (row, out) in x.chunks(cols).zip(out) {
        *out = row.iter().copied().sum::<T>() / T::from_usize(row.len());
    }
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
    use crate::{mean_cols, mean_cols_grad, mean_rows, mean_rows_grad};

    #[test]
    fn test_mean_cols() {
        #[rustfmt::skip]
        let x = [-3., 2., 3., 1.,
                            1., 5., -5., 4.,
                            -9., 2., -4., 1.];

        let mut out = [0.; 3];
        mean_cols(4, &x, &mut out);

        assert_eq!(out, [0.75, 1.25, -2.5]);
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

    #[test]
    fn test_mean_rows() {
        #[rustfmt::skip]
        let x = [-3, 2, 3, 1,
                            0, 5, -5, 4,
                            -9, 2, -4, 1];

        let mut out = [0; 4];
        mean_rows(4, &x, &mut out);

        assert_eq!(out, [-4, 3, -2, 2]);
    }

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
}
