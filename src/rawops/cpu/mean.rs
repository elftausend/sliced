use custos::prelude::Number;

use crate::sum_rows2;

pub fn mean<T: Number>(x: &[T]) -> T {
    x.iter().copied().sum::<T>() / T::from_usize(x.len())
}

pub fn mean_rows<T: Number>(cols: usize, x: &[T], out: &mut [T]) {
    sum_rows2(cols, x, out);
    for val in out {
        *val /= T::from_usize(x.len() / cols)
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
