use custos::{Buffer, Device, MainMemory, CPU};

use super::ColOp;

// TODO: shape?
impl<T: Copy, D: MainMemory> ColOp<T, (), (), D> for CPU {
    #[inline]
    fn col_op<F>(&self, cols: usize, lhs: &Buffer<T, D>, rhs: &Buffer<T, D>, f: F) -> Buffer<T>
    where
        F: Fn(T, T) -> T,
    {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));
        slice_col_op(cols, lhs, rhs, &mut out, f);
        out
    }
}

pub fn slice_col_op<T, F>(cols: usize, lhs: &[T], rhs: &[T], out: &mut [T], f: F)
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    for ((lhs_row, out), rhs) in lhs.chunks(cols).zip(out.chunks_mut(cols)).zip(rhs) {
        for (val, out) in lhs_row.iter().zip(out) {
            *out = f(*val, *rhs)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::slice_col_op;

    #[test]
    fn test_col_op() {
        #[rustfmt::skip]
        let lhs = [
            1, -1, 3, 4, 2,
            2, 3, 4, 1, 1,
            2, 0, 3, 3, -2,
        ];

        let rhs = [3, 2, 1];

        let mut out = [0; 15];

        slice_col_op(5, &lhs, &rhs, &mut out, |a, b| a + b);

        assert_eq!(out, [4, 2, 6, 7, 5, 4, 5, 6, 3, 3, 3, 1, 4, 4, -1]);
    }
}
