use custos::{Buffer, CDatatype, OpenCL, Shape};

use crate::{assign_or_set::Assign, cl_transpose, TranposeGrad};

impl<T, IS: Shape, OS: Shape> TranposeGrad<T, IS, OS> for OpenCL
where
    T: CDatatype,
    IS: Shape,
    OS: Shape,
{
    #[inline]
    fn transpose_grad(
        &self,
        rows: usize,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    ) {
        cl_transpose::<T, Assign, _, _>(self, out_grad, x_grad, rows, cols).unwrap();
    }
}
