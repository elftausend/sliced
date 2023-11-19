use custos::{Alloc, Buffer, Device, OnNewBuffer, Shape};

use crate::Matrix;

impl<'a, T, D: Device, S: Shape> From<(Buffer<'a, T, D, S>, usize, usize)> for Matrix<'a, T, D, S> {
    #[inline]
    fn from((data, rows, cols): (Buffer<'a, T, D, S>, usize, usize)) -> Self {
        Matrix { data, rows, cols }
    }
}

impl<'a, T: Copy, D: Alloc<T> + OnNewBuffer<T, D>, const N: usize>
    From<(&'a D, usize, usize, [T; N])> for Matrix<'a, T, D>
{
    #[inline]
    fn from((device, rows, cols, slice): (&'a D, usize, usize, [T; N])) -> Self {
        let data = Buffer::from((device, slice));
        Matrix { data, rows, cols }
    }
}

impl<'a, T: Copy, D: Alloc<T> + OnNewBuffer<T, D>, const N: usize>
    From<(&'a D, usize, usize, &[T; N])> for Matrix<'a, T, D>
{
    #[inline]
    fn from((device, rows, cols, slice): (&'a D, usize, usize, &[T; N])) -> Self {
        let data = Buffer::from((device, slice));
        Matrix { data, rows, cols }
    }
}

#[cfg(feature = "static-api")]
impl<'a, T: Clone> From<(usize, usize, &[T])> for Matrix<'a, T> {
    #[inline]
    fn from((rows, cols, slice): (usize, usize, &[T])) -> Self {
        Matrix::from((Buffer::from(slice), rows, cols))
    }
}

#[cfg(feature = "static-api")]
impl<'a, T: Clone, const N: usize> From<(usize, usize, [T; N])> for Matrix<'a, T> {
    #[inline]
    fn from((rows, cols, slice): (usize, usize, [T; N])) -> Self {
        Matrix::from((Buffer::from(slice), rows, cols))
    }
}

// no tuple for dims
#[cfg(not(feature = "no-std"))]
// FIXME: In this case, GraphReturn acts as an "IsDynamic" trait, as GraphReturn is not implemented for Stack
// not anymore - but the message stays the same
impl<'a, T: Copy, D: Alloc<T> + OnNewBuffer<T, D>> From<(&'a D, usize, usize, Vec<T>)>
    for Matrix<'a, T, D>
{
    fn from((device, rows, cols, data): (&'a D, usize, usize, Vec<T>)) -> Self {
        let data = Buffer::from((device, data));
        Matrix { data, rows, cols }
    }
}
