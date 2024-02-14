use std::ops::Deref;

use custos::{AddOperation, AsNoId, Buffer, Device, Retrieve, Retriever, CPU};

use super::ColOp;

// TODO: shape?
impl<T, D, Mods> ColOp<T, (), (), D> for CPU<Mods>
where
    T: Copy + 'static,
    D: Device + 'static,
    D::Base<T, ()>: Deref<Target = [T]>,
    Mods: Retrieve<Self, T> + AddOperation + 'static,
{
    #[inline]
    fn col_op<F>(
        &self,
        cols: usize,
        lhs: &Buffer<T, D>,
        rhs: &Buffer<T, D>,
        f: F,
    ) -> Buffer<T, Self>
    where
        F: Fn(T, T) -> T + Copy + 'static,
    {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));

        self.add_op(
            (cols.no_id(), lhs, rhs, &mut out, f.no_id()),
            |(cols, lhs, rhs, out, f)| {
                slice_col_op(**cols, lhs, rhs, out, **f);
                Ok(())
            },
        )
        .unwrap();
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
