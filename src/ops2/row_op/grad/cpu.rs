use core::ops::{AddAssign, Div, Mul};

use custos::prelude::One;

pub fn slice_row_op_grad_lhs<T, F>(cols: usize, lhs_grad: &mut [T], rhs: &[T], out_grad: &[T], f: F)
where
    T: Copy + One + Div<Output = T> + AddAssign + Mul<Output = T>,
    F: Fn(T) -> T,
{
    for (lhs_grad, out_grad) in lhs_grad.chunks_mut(cols).zip(out_grad.chunks(cols)) {
        for ((lhs_grad_val, rhs_val), out_grad_val) in lhs_grad.iter_mut().zip(rhs).zip(out_grad) {
            *lhs_grad_val += f(*rhs_val) * *out_grad_val;
        }
    }
}

pub fn slice_row_op_grad_rhs<T, F>(cols: usize, rhs_grad: &mut [T], lhs: &[T], out_grad: &[T], f: F)
where
    T: Copy + One + Div<Output = T> + AddAssign + Mul<Output = T>,
    F: Fn(T) -> T,
{
    for (lhs, out_grad) in lhs.chunks(cols).zip(out_grad.chunks(cols)) {
        for ((rhs_grad_val, lhs_val), out_grad_val) in rhs_grad.iter_mut().zip(lhs).zip(out_grad) {
            *rhs_grad_val += f(*lhs_val) * *out_grad_val;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{slice_row_op_grad_lhs, slice_row_op_grad_rhs};

    #[test]
    fn test_row_op_grad_mul() {
        // 2 x 3
        let lhs = [1., 2., 3., 4., 5., 6.];
        let mut lhs_grad = [0.; 6];

        let rhs = [-3., 2., 4.];
        let mut rhs_grad = [0.; 3];

        slice_row_op_grad_lhs(3, &mut lhs_grad, &rhs, &[1., 2., 3., 4., 5., 6.], |x| x);
        slice_row_op_grad_rhs(3, &mut rhs_grad, &lhs, &[1., 2., 3., 4., 5., 6.], |x| x);

        let lhs_grad_expected = [-3.0, 4.0, 12.0, -12.0, 10.0, 24.0];
        let rhs_grad_expected = [17.0, 29.0, 45.0];

        assert_eq!(lhs_grad, lhs_grad_expected);
        assert_eq!(rhs_grad, rhs_grad_expected);
    }
}
