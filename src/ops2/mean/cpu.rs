use custos::{prelude::Number, Buffer, Device, Retriever, Shape, CPU};

use crate::{slice_sum_rows2, Mean, MeanCols, MeanRows};

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

pub fn mean<T: Number>(x: &[T]) -> T {
    x.iter().copied().sum::<T>() / T::from_usize(x.len())
}

pub fn mean_rows<T: Number>(cols: usize, x: &[T], out: &mut [T]) {
    slice_sum_rows2(cols, x, out);
    for val in out {
        *val /= T::from_usize(x.len() / cols)
    }
}

pub fn mean_cols<T: Number>(cols: usize, x: &[T], out: &mut [T]) {
    for (row, out) in x.chunks(cols).zip(out) {
        *out = row.iter().copied().sum::<T>() / T::from_usize(row.len());
    }
}

#[cfg(test)]
mod tests {
    use crate::{mean_cols, mean_rows};

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
    fn test_mean_rows() {
        #[rustfmt::skip]
        let x = [-3, 2, 3, 1,
                            0, 5, -5, 4,
                            -9, 2, -4, 1];

        let mut out = [0; 4];
        mean_rows(4, &x, &mut out);

        assert_eq!(out, [-4, 3, -2, 2]);
    }
}
