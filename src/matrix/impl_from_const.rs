use custos::{prelude::Number, Alloc, Buffer, Dim2, WithShape, OnNewBuffer};

use crate::Matrix;

impl<'a, T, D, const B: usize, const A: usize> WithShape<&'a D, [[T; A]; B]>
    for Matrix<'a, T, D, Dim2<B, A>>
where
    T: Number,
    D: Alloc<T> + OnNewBuffer<T, D, Dim2<B, A>>
{
    #[inline]
    fn with(device: &'a D, array: [[T; A]; B]) -> Self {
        (Buffer::with(device, array), B, A).into()
    }
}
