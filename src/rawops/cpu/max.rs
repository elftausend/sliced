use std::fmt::Debug;

#[inline]
pub fn max<T: Ord + Copy>(x: &[T]) -> Option<T> {
    x.iter().copied().reduce(T::max)
}

pub fn max_rows<T: Ord + Copy + Debug>(cols: usize, x: &[T], out: &mut [T]) {
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