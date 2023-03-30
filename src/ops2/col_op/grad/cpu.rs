use std::ops::{AddAssign, Mul};

// TODO: Fix row_op_grad, row op only supports rhs for grad fn
pub fn slice_col_op_grad_lhs<T>(
    cols: usize,
    lhs: &[T],
    rhs: &[T],
    lhs_grad: &mut [T],
    out_grad: &[T],
    lhs_grad_fn: impl Fn(T, T) -> T,
) where
    T: Copy + AddAssign + Mul<Output = T>,
{
    let mut rhs_iter = rhs.iter();
    let mut rhs_val = rhs_iter.next().unwrap();
    for (idx, ((lhs, lhs_grad), out_grad)) in lhs.iter().zip(lhs_grad).zip(out_grad).enumerate() {
        if (idx + 1) % (cols + 1) == 0 {
            rhs_val = rhs_iter.next().unwrap();
        }

        *lhs_grad += lhs_grad_fn(*lhs, *rhs_val) * *out_grad;
    }
}

pub fn slice_col_op_grad_rhs<T>(
    cols: usize,
    lhs: &[T],
    rhs: &[T],
    rhs_grad: &mut [T],
    out_grad: &[T],
    rhs_grad_fn: impl Fn(T, T) -> T,
) where
    T: Copy + AddAssign + Mul<Output = T>,
{
    let mut rhs_idx = 0;
    for (idx, (lhs, out_grad)) in lhs.iter().zip(out_grad).enumerate() {
        if (idx +1) % (cols+1) == 0 {
            rhs_idx += 1;
        }
        rhs_grad[rhs_idx] += rhs_grad_fn(*lhs, rhs[rhs_idx]) * *out_grad;
    }
}

#[cfg(test)]
mod tests {
    use crate::{slice_col_op_grad_lhs, test_utils::roughly_equals, slice_col_op_grad_rhs};

    #[test]
    fn test_slice_col_op_grad_lhs_div() {
        #[rustfmt::skip]
        let lhs = [
            1., 2., 3., 
            4., 5., 6.
        ];

        let mut lhs_grad = [0.; 6];

        let rhs = [-3., 2.];

        slice_col_op_grad_lhs(
            3,
            &lhs,
            &rhs,
            &mut lhs_grad,
            &[1.4, 2.5, 3.3, 4., 5., 6.],
            |_, rhs| 1. / rhs,
        );

        roughly_equals(
            &lhs_grad,
            &[
                1.4 * 1. / -3.,
                2.5 * 1. / -3.,
                3.3 * 1. / -3.,
                4. * 1. / 2.,
                5. * 1. / 2.,
                6. * 1. / 2.,
            ],
        );
    }

    #[test]
    fn test_slice_col_op_grad_rhs_div() {
        #[rustfmt::skip]
        let lhs = [
            1., 2., 3., 
            4., 5., 6.
        ];


        let rhs = [-3., 2.];

        let mut rhs_grad = [0.; 2];

        let gf = |lhs: f64, rhs: f64| lhs / -(rhs * rhs);

        slice_col_op_grad_rhs(
            3,
            &lhs,
            &rhs,
            &mut rhs_grad,
            &[1.4, 2.5, 3.3, 4., 5., 6.],
            gf,
        );

        roughly_equals(
            &rhs_grad,
            &[
                gf(1., -3.) * 1.4 + gf(2., -3.) * 2.5 + gf(3., -3.) * 3.3,
                gf(4., 2.) * 4. + gf(5., 2.) * 5. + gf(6., 2.) * 6.,
            ],
        );
    }
}
