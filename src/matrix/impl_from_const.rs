use custos::{WithShape, Dim2, prelude::Number, Alloc, Buffer};

use crate::Matrix;


impl<'a, T, D, const B: usize, const A: usize> WithShape<&'a D, [[T; A]; B]>
    for Matrix<'a, T, D, Dim2<B, A>>
where
    T: Number,
    D: Alloc<'a, T, Dim2<B, A>>,
{
    #[inline]
    fn with(device: &'a D, array: [[T; A]; B]) -> Self {
        (Buffer::with(device, array), B, A).into()
    }
}