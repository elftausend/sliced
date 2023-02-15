use custos::{Buffer, Device, MainMemory, Shape, CPU};

use crate::{Max, MaxRows, MaxCols};

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

impl<T, D, IS, OS> MaxCols<T, IS, OS, D> for CPU
where
    T: Ord + Copy,
    D: MainMemory,
    IS: Shape,
    OS: Shape,
{
    #[inline]
    fn max_cols(&self, rows: usize, cols: usize, x: &Buffer<T, D, IS>) -> Buffer<T, Self, OS> {
        let mut out = self.retrieve::<_, OS>(rows);
        max_cols(cols, x, &mut out);
        out
    }
}

#[inline]
pub fn max<T: Ord + Copy>(x: &[T]) -> Option<T> {
    x.iter().copied().reduce(T::max)
}

pub fn max_rows<T: Ord + Copy>(cols: usize, x: &[T], out: &mut [T]) {
    for row in x.chunks(cols) {
        for (val, max) in row.iter().zip(out.iter_mut()) {
            *max = T::max(*val, *max)
        }
    }
}

pub fn max_cols<T: Ord + Copy>(cols: usize, x: &[T], out: &mut [T]) {
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
