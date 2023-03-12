use std::ops::AddAssign;

use custos::{Buffer, Shape, CPU};

use crate::{SumColsGrad, SumRowsGrad};

impl<T, IS, OS> SumRowsGrad<T, IS, OS> for CPU
where
    T: Copy + AddAssign,
    IS: Shape,
    OS: Shape,
{
    #[inline]
    fn sum_rows_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, OS>,
    ) {
        slice_sum_rows_grad(cols, x_grad, out_grad);
    }
}

impl<T, IS, OS> SumColsGrad<T, IS, OS> for CPU
where
    T: Copy + AddAssign,
    IS: Shape,
    OS: Shape,
{
    #[inline]
    fn sum_cols_grad(
        &self,
        cols: usize,
        x_grad: &mut Buffer<T, Self, IS>,
        out_grad: &Buffer<T, Self, IS>,
    ) {
        slice_sum_cols_grad(cols, x_grad, out_grad);
    }
}

pub fn slice_sum_rows_grad<T: Copy + AddAssign>(cols: usize, x_grad: &mut [T], out_grad: &[T]) {
    for x_grad in x_grad.chunks_mut(cols) {
        for (x, out) in x_grad.iter_mut().zip(out_grad) {
            *x += *out;
        }
        //x_grad.copy_from_slice(out_grad)
    }
}

pub fn slice_sum_cols_grad<T: Copy + AddAssign>(cols: usize, x_grad: &mut [T], out_grad: &[T]) {
    for (x_grad, out_grad) in x_grad.chunks_mut(cols).zip(out_grad) {
        for val in x_grad {
            *val += *out_grad
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{slice_sum_cols_grad, slice_sum_rows_grad};

    #[test]
    fn test_sum_cols_grad() {
        #[rustfmt::skip]
        let _x = [
            1, 2, 3,
            3, 2, 1,
            2, 3, 1,
            4, 8, 1
        ];
        let mut x_grad = [0; 12];
        slice_sum_cols_grad(3, &mut x_grad, &[2, 3, 4, -1]);

        #[rustfmt::skip]
        let expected = [
            2, 2, 2, 
            3, 3, 3, 
            4, 4, 4, 
            -1, -1, -1
        ];
        assert_eq!(x_grad, expected);
    }

    #[test]
    fn test_sum_rows_grad() {
        #[rustfmt::skip]
        let _x = [
            1, 2, 3,
            3, 2, 1,
            2, 3, 1,
            4, 8, 1
        ];
        let mut x_grad = [0; 12];
        slice_sum_rows_grad(3, &mut x_grad, &[2, 4, -1]);

        #[rustfmt::skip]
        let expected = [
            2, 4, -1, 
            2, 4, -1, 
            2, 4, -1, 
            2, 4, -1
        ];
        assert_eq!(x_grad, expected);
    }
}
