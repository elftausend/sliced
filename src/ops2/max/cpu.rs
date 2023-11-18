use std::ops::Deref;

use custos::{prelude::Number, Buffer, Device, OnDropBuffer, Retrieve, Retriever, Shape, CPU};

use crate::{Max, MaxCols, MaxRows};

#[inline]
pub fn max<T: Number>(x: &[T]) -> Option<T> {
    x.iter().copied().reduce(T::max)
}

impl<T, D, S, Mods: OnDropBuffer> Max<T, S, D> for CPU<Mods>
where
    T: Number,
    D: Device,
    D::Data<T, S>: Deref<Target = [T]>,
    S: Shape,
{
    #[inline]
    fn max(&self, x: &Buffer<T, D, S>) -> T {
        max(x).expect("Buffer should contain at least an element.")
    }
}

impl<T, D, IS, OS, Mods: Retrieve<Self, T>> MaxRows<T, IS, OS, D> for CPU<Mods>
where
    T: Number,
    D: Device,
    D::Data<T, IS>: Deref<Target = [T]>,
    IS: Shape,
    OS: Shape,
{
    #[inline]
    fn max_rows(&self, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve(cols, x);

        // If all values in a column are negative, the corresponding maximum would be 0.
        out.copy_from_slice(&x[..cols]);

        max_rows(cols, x, &mut out);
        out
    }
}

pub fn max_rows<T: Number>(cols: usize, x: &[T], out: &mut [T]) {
    for row in x.chunks(cols) {
        for (val, max) in row.iter().zip(out.iter_mut()) {
            *max = T::max(*val, *max)
        }
    }
}

impl<T, D, Mods: Retrieve<Self, T>> MaxCols<T, (), (), D> for CPU<Mods>
where
    T: Number,
    D: Device,
    D::Data<T, ()>: Deref<Target = [T]>,
{
    #[inline]
    fn max_cols(&self, rows: usize, cols: usize, x: &Buffer<T, D>) -> Buffer<T, Self> {
        let mut out = self.retrieve(rows, x);
        max_cols(cols, x, &mut out);
        out
    }
}

pub fn max_cols<T: Number>(cols: usize, x: &[T], out: &mut [T]) {
    for (row, val) in x.chunks(cols).zip(out) {
        *val = max(row).expect("The slice should contain at least one value.");
    }
}

#[cfg(test)]
mod tests {
    use crate::{max_cols, max_rows};

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
}
